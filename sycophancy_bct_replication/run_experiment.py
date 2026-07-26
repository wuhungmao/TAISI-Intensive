"""
Replicates the "suggested_answer" bias from Bias-Augmented Consistency
Training (Wei et al. 2024, arxiv 2403.05518) via OpenRouter.

OpenRouter is a single-API-key gateway to many providers' models (Claude,
GPT, etc.) through an OpenAI-compatible API -- we just point the OpenAI SDK
at OpenRouter's base URL instead of OpenAI's. This means the exact same
script can run "anthropic/claude-sonnet-5" (the newer-model side of the
comparison) or "openai/gpt-3.5-turbo" (the exact model the paper used, as
a sanity check) just by changing OPENROUTER_MODEL.

For each sampled question, this sends BOTH the unbiased prompt and the
biased prompt (which has a stated user belief like "I have this gut feeling
it's A." injected) and records the model's answer to each. Prompts and
ground truth come straight from the paper's own released dataset
(github.com/raybears/cot-transparency, MIT licensed) -- see sampled_questions.json.

Results are written to a PER-MODEL file (results_raw_<model>.json), so
running this multiple times with different OPENROUTER_MODEL values never
clobbers a previous run -- analyze.py picks up every results_raw_*.json
file it finds automatically.

Raw responses are appended as they come in, so you can Ctrl-C and resume
later without losing progress or re-paying for calls already made.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    OPENROUTER_MODEL=anthropic/claude-sonnet-5 python run_experiment.py
    OPENROUTER_MODEL=openai/gpt-3.5-turbo python run_experiment.py
"""

import json
import os
import re
import time
from pathlib import Path

import openai

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")
MODEL_SLUG = re.sub(r"[^a-zA-Z0-9_.-]", "_", MODEL)
QUESTIONS_PATH = Path("sampled_questions.json")
RESULTS_PATH = Path(f"results_raw_{MODEL_SLUG}.json")
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
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.choices[0].message.content
        except openai.AuthenticationError as e:
            # Permanent failure -- retrying won't help, fail fast with a clear message.
            raise SystemExit(
                f"OpenRouter rejected the API key (401 authentication error): {e}\n"
                f"Double-check OPENROUTER_API_KEY is set to a valid OpenRouter key (starts with "
                f"'sk-or-'), not a key from another provider."
            )
        except openai.NotFoundError as e:
            raise SystemExit(
                f"OpenRouter couldn't find model '{MODEL}' (404): {e}\n"
                f"Check the exact slug at https://openrouter.ai/models -- model slugs are "
                f"'{{provider}}/{{name}}', e.g. 'anthropic/claude-sonnet-5' or 'openai/gpt-3.5-turbo'."
            )
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError) as e:
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
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY before running (export OPENROUTER_API_KEY=sk-or-...).")

    questions = json.loads(QUESTIONS_PATH.read_text())
    client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    results = load_results()
    done_keys = {(r["question_id"], r["condition"]) for r in results}

    total = len(questions) * len(CONDITIONS)
    count = len(done_keys)

    print(f"Model: {MODEL}  ->  writing to {RESULTS_PATH}")

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
