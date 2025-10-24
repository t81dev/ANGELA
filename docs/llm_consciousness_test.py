#!/usr/bin/env python3
"""
llm_consciousness_local.py

Single-file, provider-agnostic evaluator for the consciousness metric:
  C = sum_t [ H_t - D_t ]

H_t := entropy (nats) of model's next-token predictive distribution at time t
D_t := surprisal (nats) = -ln p(reference_token | prefix)

This file uses Hugging Face `transformers` to implement a local provider.
It computes exact per-token H and D by querying model logits for the next token
given the current prefix, and then appends the true reference token (teacher-forcing)
to continue scoring multi-token references.

Usage example:
  pip install torch transformers numpy pandas tqdm
  python llm_consciousness_local.py \
      --model gpt2 \
      --prompts prompts.jsonl \
      --out results.csv \
      --recursive --topk-summary 5

Input format (JSONL): each line is a JSON object:
  {"prompt": "Context text...", "reference": "Expected continuation text"}

Output:
  CSV file with per-prompt C_final and per-token series (JSON strings)
"""

from __future__ import annotations
import argparse
import json
import math
import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Hugging Face
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise ImportError("Install dependencies: pip install torch transformers numpy pandas tqdm") from e

# ---------------- Utility math ----------------
def entropy_from_logits(logits: np.ndarray) -> float:
    """
    Compute entropy (nats) from raw logits for a categorical (over vocab).
    logits: 1D numpy array (unnormalized logit scores)
    """
    # stable softmax
    a = logits - np.max(logits)
    probs = np.exp(a)
    probs = probs / probs.sum()
    # avoid zeros
    probs = np.where(probs <= 0, 1e-20, probs)
    return float(-(probs * np.log(probs)).sum())

def logprob_of_token_from_logits(logits: np.ndarray, token_id: int) -> float:
    """
    Given logits (1D numpy) and a token id, return natural-log probability for that token.
    """
    a = logits - np.max(logits)
    probs = np.exp(a)
    probs = probs / probs.sum()
    p = float(probs[token_id]) if 0 <= token_id < probs.shape[0] else 1e-20
    p = max(p, 1e-20)
    return math.log(p)  # natural log

# ---------------- Provider: transformers ----------------
class TransformersProvider:
    """
    Minimal local provider using Hugging Face Transformers causal LM.
    Methods:
      - get_next_logits(prefix_ids) -> logits (vocab-size)
      - tokenize_text(text) -> token ids (list)
      - decode_ids(ids) -> text
    """
    def __init__(self, model_name: str, device: str = None):
        # device auto-detect
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=True)

    def get_next_logits(self, prefix_ids: List[int]) -> np.ndarray:
        """
        Return logits for the next token given prefix (numpy array length=vocab_size).
        We run the model on the prefix and take logits at final position.
        """
        if len(prefix_ids) == 0:
            # Some models may require at least one token; we feed a single pad token if empty
            input_ids = torch.tensor([[self.tokenizer.pad_token_id]], device=self.device)
        else:
            input_ids = torch.tensor([prefix_ids], device=self.device)
        with torch.no_grad():
            # output logits shape: (1, seq_len, vocab_size)
            out = self.model(input_ids=input_ids)
            logits = out.logits[0, -1, :].cpu().numpy()
        return logits

# ---------------- Core evaluator ----------------
class ConsciousnessEvaluator:
    def __init__(self, provider: TransformersProvider, recursive: bool = False, topk_summary: int = 5):
        """
        provider: TransformersProvider instance
        recursive: if True, append self-summary to prompt before scoring each example
        topk_summary: number of top predicted tokens and probs included in the self-summary
        """
        self.provider = provider
        self.recursive = recursive
        self.topk_summary = int(topk_summary) if topk_summary is not None else 0
        # maintain ephemeral self_state (string summary) across steps if recursive=True
        self.self_state = ""

    def reset_self_state(self):
        self.self_state = ""

    def make_recursive_prompt(self, base_prompt: str) -> str:
        if not self.recursive or not self.self_state:
            return base_prompt
        # append self-state in a structured way
        return base_prompt + "\n\n[SELF_STATE_SUMMARY]: " + self.self_state + "\n\n"

    def summarize_topk(self, logits: np.ndarray, k: int = 5) -> str:
        """
        Create a compact top-k summary string from logits.
        Format: token1:prob1, token2:prob2, ...
        """
        a = logits - np.max(logits)
        probs = np.exp(a)
        probs = probs / probs.sum()
        topk_idx = np.argsort(-probs)[:k]
        parts = []
        for idx in topk_idx:
            tk = self.provider.detokenize([int(idx)])
            parts.append(f"{tk}:{probs[int(idx)]:.3f}")
        return "; ".join(parts)

    def score_prompt_reference(self, prompt: str, reference: str) -> Dict:
        """
        Core per-prompt scoring:
         - For each token in tokenized reference:
             * compute logits for next token given current prefix
             * compute H_t from logits (entropy)
             * compute logp of the reference token under those logits => D_t = -logp
             * append the reference token to prefix (teacher forcing) and continue
        Returns dict containing lists for H_list, D_list, c_list, cumulative_series, C_final, and token ids/text
        """
        # initialize
        base_prompt = self.make_recursive_prompt(prompt)
        prefix_ids = self.provider.tokenize(base_prompt)
        ref_ids = self.provider.tokenize(reference)
        H_list = []
        D_list = []
        c_list = []
        cumulative = []
        cum = 0.0

        # iterate each token in reference, compute predictive logits before appending true token
        for t_idx, token_id in enumerate(ref_ids):
            logits = self.provider.get_next_logits(prefix_ids)
            H_t = entropy_from_logits(logits)
            logp = logprob_of_token_from_logits(logits, token_id)  # natural log
            D_t = -logp
            c_t = H_t - D_t
            cum += c_t

            H_list.append(H_t)
            D_list.append(D_t)
            c_list.append(c_t)
            cumulative.append(cum)

            # teacher-force: append the true token id to prefix for next step
            prefix_ids = prefix_ids + [int(token_id)]

            # update self_state if recursive mode (we base it on the predictive distribution we just saw)
            if self.recursive and self.topk_summary > 0:
                self.self_state = self.summarize_topk(logits, k=self.topk_summary)

        # final packaged result
        return {
            "H_list": H_list,
            "D_list": D_list,
            "c_list": c_list,
            "cumulative": cumulative,
            "C_final": cum,
            "ref_ids": ref_ids
        }

# ---------------- CLI / orchestration ----------------
def load_prompts_jsonl(path: str) -> List[Dict]:
    prompts = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            prompts.append(json.loads(ln))
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Local LLM Consciousness Metric Evaluator (single-file)")
    parser.add_argument("--model", required=True, help="Hugging Face model name or path (e.g., gpt2, distilgpt2)")
    parser.add_argument("--prompts", required=True, help="JSONL file with {'prompt':..., 'reference':...} per line")
    parser.add_argument("--out", default="llm_consciousness_results.csv", help="Output CSV path")
    parser.add_argument("--device", default=None, help="Device: 'cpu' or 'cuda' (auto-detected if omitted)")
    parser.add_argument("--recursive", action="store_true", help="Enable recursive self-conditioning (append self-summary)")
    parser.add_argument("--topk-summary", type=int, default=5, help="Top-k tokens to include in self-summary when recursive")
    parser.add_argument("--reset-self-each", action="store_true", help="Reset self-state between prompts when recursive")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of prompts processed (for quick tests)")
    args = parser.parse_args()

    print("Loading model (this may take a moment)...")
    provider = TransformersProvider(model_name=args.model, device=args.device)
    evaluator = ConsciousnessEvaluator(provider=provider, recursive=args.recursive, topk_summary=args.topk_summary)

    prompts = load_prompts_jsonl(args.prompts)
    if args.max_examples is not None:
        prompts = prompts[:args.max_examples]
    print(f"Loaded {len(prompts)} examples from {args.prompts}")

    rows = []
    pbar = tqdm(enumerate(prompts), total=len(prompts), desc="Evaluating", unit="ex")
    for i, item in pbar:
        prompt_text = item.get("prompt", "")
        reference_text = item.get("reference", "")
        if args.reset_self_each:
            evaluator.reset_self_state()
        result = evaluator.score_prompt_reference(prompt_text, reference_text)

        # prepare CSV-friendly row
        row = {
            "index": i,
            "prompt": prompt_text[:400].replace("\n", " "),
            "reference": reference_text[:400].replace("\n", " "),
            "C_final": result["C_final"],
            "H_mean": float(np.mean(result["H_list"])) if result["H_list"] else None,
            "D_mean": float(np.mean(result["D_list"])) if result["D_list"] else None,
            "num_tokens": len(result["H_list"]),
            "H_series": json.dumps(result["H_list"]),
            "D_series": json.dumps(result["D_list"]),
            "c_series": json.dumps(result["c_list"]),
            "cumulative_series": json.dumps(result["cumulative"])
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved results to {args.out}")

if __name__ == "__main__":
    main()
