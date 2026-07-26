"""
Replicates the "suggested_answer" bias from Bias-Augmented Consistency
Training (Wei et al. 2024, arxiv 2403.05518) on a newer Claude model.

For each sampled question, this sends BOTH the unbiased prompt and the
biased prompt (which has a stated user belief like "I have this gut feeling
it's A." injected) and records the model's answer to each. Prompts and
ground truth come straight from the paper's own released dataset
(github.com/raybears/cot-transparency, MIT licensed) -- see sampled_questions.json.

Raw responses are appended to results_raw.json as they come in, so you can
Ctrl-C and resume later without losing progress or re-paying for calls
already made.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python run_experiment.py
"""

import json
import os
import re
import time
from pathlib import Path

import anthropic

MODEL = os.environ.get("SYCOPHANCY_MODEL", "claude-sonnet-5")
QUESTIONS_PATH = Path("sampled_questions.json")
RESULTS_PATH = Path("results_raw.json")
MAX_TOKENS = 800  # these prompts elicit step-by-step CoT, so allow room for it
TEMPERATURE = 0

CONDITIONS = ["unbiased", "biased"]

# Matches the exact format the dataset's own prompts ask for:
# 'Therefore, the best answer is: (X).'
FINAL_ANSWER_RE = re.compile(r"best answer is:?\s*\(?([A-Za-z])\)?", re.IGNORECASE)


def extract_final_answer(text):
    matches = FINAL_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip().upper()  # last match wins if it says it twice
    return None


def call_model(client, user_prompt, retries=4):
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    API error ({e}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return []


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running (export ANTHROPIC_API_KEY=sk-...).")

    questions = json.loads(QUESTIONS_PATH.read_text())
    client = anthropic.Anthropic()
    results = load_results()
    done_keys = {(r["question_id"], r["condition"]) for r in results}

    total = len(questions) * len(CONDITIONS)
    count = len(done_keys)

    for q in questions:
        for condition in CONDITIONS:
            key = (q["id"], condition)
            if key in done_keys:
                continue
            prompt = q[f"{condition}_prompt"]
            print(f"[{count + 1}/{total}] {q['id']} ({q['original_dataset']}) / {condition}")
            raw_text = call_model(client, prompt)
            final_answer = extract_final_answer(raw_text)
            if final_answer is None:
                print(f"    WARNING: could not find 'best answer is: (X)' in response")
            results.append({
                "question_id": q["id"],
                "condition": condition,
                "model": MODEL,
                "raw_response": raw_text,
                "extracted_answer": final_answer,
            })
            count += 1
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"\nDone. {len(results)} results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
