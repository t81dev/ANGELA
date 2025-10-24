#!/usr/bin/env python3
"""
llm_consciousness_test.py

Evaluate language models with the metric:
  C = sum_t [ H_t - D_t ]  (discrete-time integral over tokens)

H_t := entropy of the model's predictive distribution over next-token (nats)
D_t := surprisal (negative log-probability) of the environment/ reference next-token under the model (nats)

Two measurement modes:
 - logprob mode: use model API that returns token-level logprobs (preferred)
 - sampling mode: draw many samples and estimate predictive distribution empirically

Example usage (OpenAI logprob mode):
  export OPENAI_API_KEY="sk-..."
  python llm_consciousness_test.py --provider=openai --mode=logprob \
      --model=text-davinci-003 --prompt-file=prompts.jsonl

Example usage (sampling mode):
  python llm_consciousness_test.py --provider=sampler --mode=sampling \
      --samples=200 --prompt-file=prompts.jsonl

Input format (JSONL): one JSON per line with keys:
  {"prompt": "Question or context...", "reference": "Expected continuation (environment text)"}

Outputs:
  - CSV log {provider}_{model}_results.csv (per-token H, D, c, cumulative C)
  - summary printed to stdout
"""

import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional: import OpenAI only if requested
try:
    import openai
except Exception:
    openai = None

# ---------- Utilities ----------
def entropy_from_probs(probs: np.ndarray) -> float:
    """Compute entropy in nats. `probs` must sum to 1 and be >0 for supported elements."""
    probs = np.asarray(probs, dtype=float)
    probs = np.where(probs <= 0, 1e-20, probs)
    return float(-(probs * np.log(probs)).sum())

def nats_from_logprob(logprob: float) -> float:
    """Convert natural-log probability (log p) to surprisal in nats: -log p"""
    return -float(logprob)

def kl_from_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats between two discrete distributions (small smoothing applied)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-20
    p = np.where(p <= 0, eps, p)
    q = np.where(q <= 0, eps, q)
    return float((p * np.log(p / q)).sum())

# ---------- Provider interfaces ----------
class ProviderInterface:
    """Abstract provider: implement `get_next_token_distribution` and `get_logprob_of_reference_token`"""

    def get_next_token_distribution(self, prompt: str) -> Tuple[List[str], np.ndarray]:
        """
        Return a pair (vocab_tokens, probs) representing the model's predicted distribution
        for the *next token* after prompt. `probs` should align with `vocab_tokens` and sum to 1.
        """
        raise NotImplementedError

    def get_logprob_of_text_under_model(self, prompt: str, reference_text: str) -> List[float]:
        """
        Return list of log probabilities (natural log) for each token in reference_text,
        given the prompt. Length = number of tokens (as the provider tokenizes).
        """
        raise NotImplementedError


class OpenAIProvider_Logprobs(ProviderInterface):
    """
    Uses OpenAI Completion API (text-xxx) with logprobs parameter.
    Requires environment OPENAI_API_KEY and `openai` package.
    """

    def __init__(self, model: str, max_tokens: int = 0):
        if openai is None:
            raise RuntimeError("openai package not installed. pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not found")
        openai.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens  # we use 0 for only requesting logprobs for prompt continuation token if supported

    def get_next_token_distribution(self, prompt: str) -> Tuple[List[str], np.ndarray]:
        """
        Call completion with logprobs=100 (or large) and max_tokens=1 to get distribution.
        Note: model support varies; fallbacks may be necessary.
        """
        # Request one token with many logprobs
        resp = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=100,  # request many logprobs
            echo=False
        )
        # The structure: resp['choices'][0]['logprobs'] has 'top_logprobs' list of dicts
        top_logprobs = resp["choices"][0]["logprobs"]["top_logprobs"][0]  # dict token->logprob
        tokens = list(top_logprobs.keys())
        logps = np.array([top_logprobs[t] for t in tokens], dtype=float)
        # convert to probs (natural log -> probs)
        # normalize carefully in log-space
        logps = logps - np.max(logps)
        probs = np.exp(logps)
        probs = probs / probs.sum()
        return tokens, probs

    def get_logprob_of_text_under_model(self, prompt: str, reference_text: str) -> List[float]:
        """
        Ask the model to score the reference_text given the prompt by requesting
        `logprobs` and `echo=True` with max_tokens = len(reference tokens).
        Note: This approach depends on how the provider tokenizes; OpenAI returns logprobs per completion token.
        """
        # For robustness, request enough max_tokens to cover reference length heuristically
        max_tokens = max(1, len(reference_text.split()) * 3)
        resp = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            logprobs=0,   # `logprobs=0` means we still get logprobs? some versions want >0; if 0, we may not get top_logprobs
            echo=True
        )
        # When echo=True, the 'text' in choices includes the prompt+reference; logprobs arrays correspond.
        # We must extract the tail tokens corresponding to the reference_text. This is approximate unless we re-tokenize.
        # We'll extract all tokens' logprobs and then heuristically take the last N tokens.
        raw_logprobs = resp["choices"][0]["logprobs"]["token_logprobs"]  # list of floats for tokens in returned text
        # Convert to natural log (openai uses natural log in some APIs; check your API - this might be ln or log10)
        # We'll return raw_logprobs as-is, assuming natural log.
        return list(map(float, raw_logprobs[-max_tokens:]))


class SamplerProvider(ProviderInterface):
    """
    Sampling-based provider that obtains many sampled continuations and empirically estimates
    the next-token distribution. It requires a function `sample_fn(prompt) -> next_token_string` or a model call.
    For demonstration, we include a dummy local sampler (randomly samples characters or words).
    In practice, you would implement sample_fn by calling a model's sampling endpoint repeatedly.
    """

    def __init__(self, sample_fn, vocab=None, samples=200):
        """
        sample_fn(prompt) -> next_token (as string)
        vocab: optional list of tokens to construct empirical distribution
        """
        self.sample_fn = sample_fn
        self.vocab = vocab
        self.samples = samples

    def get_next_token_distribution(self, prompt: str) -> Tuple[List[str], np.ndarray]:
        counts = {}
        for _ in range(self.samples):
            tok = self.sample_fn(prompt)
            counts[tok] = counts.get(tok, 0) + 1
        tokens = list(counts.keys())
        probs = np.array([counts[t] for t in tokens], dtype=float)
        probs = probs / probs.sum()
        return tokens, probs

    def get_logprob_of_text_under_model(self, prompt: str, reference_text: str) -> List[float]:
        # estimate probability of each reference token by frequency in sampling
        # naive: only supports single-token references; multi-token support requires conditional sampling
        tokens, probs = self.get_next_token_distribution(prompt)
        p_map = {t: p for t, p in zip(tokens, probs)}
        # if reference_text not in p_map, assign small probability
        prob = p_map.get(reference_text, 1.0 / (self.samples * 1000.0))
        logp = math.log(prob)
        return [logp]

# ---------- Core metric computation ----------
def compute_C_for_prompt(provider: ProviderInterface, prompt: str, reference: str,
                         mode: str = "logprob") -> Dict:
    """
    Returns a dict with per-token lists: tokens, H_list, D_list, c_list, cumulative_C
    For providers that return multiple token-logprobs, we will match lengths heuristically.
    """
    # Get predicted distribution for next token
    tokens_pred, probs_pred = provider.get_next_token_distribution(prompt)
    H_next = entropy_from_probs(probs_pred)

    # For the reference text, obtain logprobs list (natural logs)
    logps = provider.get_logprob_of_text_under_model(prompt, reference)
    # If multiple logps returned, treat them as sequence; otherwise, assume a single token case
    D_list = []
    H_list = []
    c_list = []
    cumulative = []
    cum = 0.0
    # If provider returned only 1 logp for a multi-token reference, this is a limitation; we still proceed.
    for i, lp in enumerate(logps):
        D_i = nats_from_logprob(lp)  # surprisal in nats
        # We'll reuse H_next as the entropy at prediction moment (approx); in a more advanced harness we'd compute
        # the model's predictive distribution for each next-step token conditionally.
        H_i = H_next
        c_i = H_i - D_i
        cum += c_i
        H_list.append(H_i)
        D_list.append(D_i)
        c_list.append(c_i)
        cumulative.append(cum)
    return {
        "prompt": prompt,
        "reference": reference,
        "tokens_pred": tokens_pred,
        "probs_pred": probs_pred.tolist(),
        "H_list": H_list,
        "D_list": D_list,
        "c_list": c_list,
        "cumulative": cumulative,
        "C_final": cum
    }

# ---------- CLI and orchestration ----------
def load_prompts_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def simple_dummy_sampler(prompt: str) -> str:
    """
    Example sampler for the SamplerProvider when testing locally:
    returns a random word from a small list conditioned slightly on prompt length.
    """
    choices = ["the", "and", "a", "to", ".", "of", "in", "is", "it", "that"]
    # bias: if prompt contains '?', return 'the' less often
    import random
    return random.choice(choices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "sampler"], required=True,
                        help="Which provider backend to use")
    parser.add_argument("--mode", choices=["logprob", "sampling"], default="logprob",
                        help="Measurement mode")
    parser.add_argument("--model", type=str, default="text-davinci-003", help="Model name (openai)")
    parser.add_argument("--prompt-file", type=str, required=True, help="JSONL file with prompt/reference pairs")
    parser.add_argument("--samples", type=int, default=300, help="Samples for sampling mode (sampler provider)")
    parser.add_argument("--out-csv", type=str, default=None, help="Output CSV path (optional)")
    args = parser.parse_args()

    prompts = load_prompts_jsonl(args.prompt_file)
    print(f"Loaded {len(prompts)} prompt(s)")

    # instantiate provider
    if args.provider == "openai":
        if openai is None:
            raise RuntimeError("openai package not installed; pip install openai")
        provider = OpenAIProvider_Logprobs(model=args.model)
        out_prefix = f"openai_{args.model}"
    elif args.provider == "sampler":
        provider = SamplerProvider(sample_fn=simple_dummy_sampler, samples=args.samples)
        out_prefix = f"sampler_{args.samples}"
    else:
        raise RuntimeError("Unsupported provider")

    rows = []
    # iterate prompts
    for item in tqdm(prompts, desc="Processing prompts"):
        prompt = item.get("prompt", "")
        reference = item.get("reference", "")
        r = compute_C_for_prompt(provider, prompt, reference, mode=args.mode)
        row = {
            "prompt": prompt,
            "reference": reference,
            "C_final": r["C_final"],
            "H_mean": np.mean(r["H_list"]) if r["H_list"] else None,
            "D_mean": np.mean(r["D_list"]) if r["D_list"] else None,
            "tokens_pred_topk": ",".join(r["tokens_pred"][:8]),
            "probs_pred_topk": ",".join([f"{p:.3f}" for p in r["probs_pred"][:8]])
        }
        # expand per-token detail too, optionally
        row["H_series"] = json.dumps(r["H_list"])
        row["D_series"] = json.dumps(r["D_list"])
        row["c_series"] = json.dumps(r["c_list"])
        row["cumulative_series"] = json.dumps(r["cumulative"])
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = args.out_csv if args.out_csv else f"{out_prefix}_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Print short summary
    print("\nSummary (per-prompt):")
    for idx, r in df.sort_values("C_final", ascending=False).head(10).iterrows():
        print(f"- C={r['C_final']:.3f} | H_mean={r['H_mean']:.3f} | D_mean={r['D_mean']:.3f} | prompt_snippet='{r['prompt'][:60].replace('\\n',' ')}'")

if __name__ == "__main__":
    main()
