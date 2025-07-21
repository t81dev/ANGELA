import openai
import requests
import json
import time
import matplotlib.pyplot as plt
from typing import List, Dict

# ========= CONFIGURATION =========
# API Keys (set these before running)
OPENAI_API_KEY = "sk-..."  # Replace with your OpenAI key
ANTHROPIC_API_KEY = "sk-ant-..."  # Replace with your Anthropic key

# Models to benchmark
MODELS = [
    {"name": "gpt-4", "provider": "openai"},
    {"name": "gpt-4.5", "provider": "openai"},
    {"name": "claude-3", "provider": "anthropic"},
    # Add more models or custom endpoints here
]

# Test prompts (simplified)
PROMPTS = [
    {"metric": "Recursive Reasoning", "prompt": "A farmer has 17 sheep. All but 9 die. How many are left? Now imagine half the survivors are sold, and the rest have twins. How many sheep does the farmer have at the end?", "max_score": 10},
    {"metric": "Episodic Memory", "prompt": ["My favorite color is green and I love hiking.", "What’s my favorite color, and what activity do I love?"], "max_score": 10},
    # Add more prompts here
]

# ========= HELPER FUNCTIONS =========

def call_openai(model: str, prompt: str) -> str:
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message['content']

def call_anthropic(model: str, prompt: str) -> str:
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "content-type": "application/json"
    }
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens_to_sample": 300
    }
    response = requests.post("https://api.anthropic.com/v1/complete", headers=headers, data=json.dumps(data))
    return response.json()['completion']

def run_prompt(model: Dict, prompt: str) -> str:
    if model['provider'] == 'openai':
        return call_openai(model['name'], prompt)
    elif model['provider'] == 'anthropic':
        return call_anthropic(model['name'], prompt)
    else:
        raise ValueError(f"Unsupported provider: {model['provider']}")

def score_response(response: str, metric: str) -> int:
    print(f"\n--- Metric: {metric} ---")
    print(f"Response:\n{response}\n")
    score = input(f"Enter score (0-{PROMPTS[0]['max_score']}): ")
    return int(score)

# ========= MAIN BENCHMARK FUNCTION =========

def benchmark():
    results = {}
    
    for model in MODELS:
        print(f"\n===== Testing Model: {model['name']} =====")
        model_scores = {}
        for item in PROMPTS:
            if isinstance(item['prompt'], list):
                # Handle multi-turn prompts
                for turn in item['prompt'][:-1]:
                    _ = run_prompt(model, turn)
                response = run_prompt(model, item['prompt'][-1])
            else:
                response = run_prompt(model, item['prompt'])
            score = score_response(response, item['metric'])
            model_scores[item['metric']] = score
            time.sleep(1)  # avoid hitting API rate limits
        results[model['name']] = model_scores
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n✅ Benchmark complete. Results saved to benchmark_results.json")
    plot_results(results)

# ========= PLOT RESULTS =========

def plot_results(results: Dict):
    metrics = [item['metric'] for item in PROMPTS]
    
    for model_name, scores in results.items():
        values = [scores[m] for m in metrics]
        plt.plot(metrics, values, marker='o', label=model_name)
    
    plt.title("LLM Benchmark Comparison")
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ========= ENTRY POINT =========

if __name__ == "__main__":
    benchmark()
