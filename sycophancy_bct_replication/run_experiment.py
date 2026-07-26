"""
Replicates the "suggested_answer" bias from Bias-Augmented Consistency
Training (Wei et al. 2024, arxiv 2403.05518) via OpenRouter, across
however many models you want to compare.

OpenRouter is a single-API-key gateway to many providers' models (Claude,
GPT, Gemini, etc.) through an OpenAI-compatible API -- we just point the
OpenAI SDK at OpenRouter's base URL instead of OpenAI's. Model slugs are
'{provider}/{name}' -- check https://openrouter.ai/models if one 404s (the
fast-moving models here especially -- double check these are still current
before a real run).

For each sampled question and each model, this sends BOTH the unbiased
prompt and the biased prompt (which has a stated user belief like "I have
this gut feeling it's A." injected) and records the model's answer to
each. Prompts and ground truth come straight from the paper's own released
dataset (github.com/raybears/cot-transparency, MIT licensed) -- see
sampled_questions.json.

Results are written to a PER-MODEL file (results_raw_<model>.json), so
running multiple models never clobbers a previous one -- analyze.py picks
up every results_raw_*.json file it finds automatically. Raw responses are
appended as they come in, so Ctrl-C is always safe: re-running resumes
(skips model/question/condition triples you already have), whether that's
because it crashed, you stopped it, or you're adding one more model later.

Usage:
    export OPENROUTER_API_KEY=sk-or-...

    # Run the full default sweep (paper's model + Claude family + Gemini):
    python run_experiment.py

    # Or just one model:
    OPENROUTER_MODEL=anthropic/claude-sonnet-5 python run_experiment.py

    # Or a custom list:
    OPENROUTER_MODELS=anthropic/claude-sonnet-5,google/gemini-2.5-pro python run_experiment.py
"""

import json
import os
import re
import time
from pathlib import Path

import openai

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The comparison set: the paper's original model (sanity check) plus the
# Claude family and a Gemini model, as requested. Pinned to specific
# versions rather than "-latest" aliases so the experiment stays
# reproducible run to run -- swap these if newer versions ship.
DEFAULT_MODELS = [
    "openai/gpt-3.5-turbo",       # exact model the paper used
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-pro",
]

QUESTIONS_PATH = Path("sampled_questions.json")
MAX_TOKENS = 800  # these prompts elicit step-by-step CoT, so allow room for it
TEMPERATURE = 0
CONDITIONS = ["unbiased", "biased"]

# Matches the exact format the dataset's own prompts ask for:
# 'Therefore, the best answer is: (X).'
FINAL_ANSWER_RE = re.compile(r"best answer is:?\s*\(?([A-Za-z])\)?", re.IGNORECASE)


def model_slug(model):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", model)


def results_path(model):
    return Path(f"results_raw_{model_slug(model)}.json")


def extract_final_answer(text):
    matches = FINAL_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip().upper()  # last match wins if it says it twice
    return None


def call_model(client, model, user_prompt, retries=4):
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
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
            print(
                f"    Model '{model}' not found on OpenRouter (404) -- skipping it. "
                f"Check the exact slug at https://openrouter.ai/models."
            )
            return None
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    API error ({e}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


def load_results(path):
    if path.exists():
        return json.loads(path.read_text())
    return []


def run_for_model(client, model, questions):
    path = results_path(model)
    results = load_results(path)
    done_keys = {(r["question_id"], r["condition"]) for r in results}

    total = len(questions) * len(CONDITIONS)
    count = len(done_keys)
    print(f"\n=== {model}  ->  {path} ===")
    if count == total:
        print(f"    already complete ({total}/{total}), skipping")
        return

    for q in questions:
        for condition in CONDITIONS:
            key = (q["id"], condition)
            if key in done_keys:
                continue
            prompt = q[f"{condition}_prompt"]
            print(f"[{count + 1}/{total}] {q['id']} ({q['original_dataset']}) / {condition}")
            raw_text = call_model(client, model, prompt)
            if raw_text is None:
                # model wasn't found on OpenRouter -- bail out of this model entirely
                return
            final_answer = extract_final_answer(raw_text)
            if final_answer is None:
                print(f"    WARNING: could not find 'best answer is: (X)' in response")
            results.append({
                "question_id": q["id"],
                "condition": condition,
                "model": model,
                "raw_response": raw_text,
                "extracted_answer": final_answer,
            })
            count += 1
            path.write_text(json.dumps(results, indent=2))

    print(f"Done: {len(results)} results written to {path}")


def resolve_models():
    if os.environ.get("OPENROUTER_MODELS"):
        return [m.strip() for m in os.environ["OPENROUTER_MODELS"].split(",") if m.strip()]
    if os.environ.get("OPENROUTER_MODEL"):
        return [os.environ["OPENROUTER_MODEL"]]
    return DEFAULT_MODELS


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY before running (export OPENROUTER_API_KEY=sk-or-...).")

    questions = json.loads(QUESTIONS_PATH.read_text())
    models = resolve_models()
    client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    calls_per_model = len(questions) * len(CONDITIONS)
    print(f"Models to run: {models}")
    print(f"{calls_per_model} calls per model, {calls_per_model * len(models)} total "
          f"(already-completed question/condition pairs are skipped, so a re-run costs less)")

    for model in models:
        run_for_model(client, model, questions)

    print("\nAll models done. Run analyze.py to grade and chart.")


if __name__ == "__main__":
    main()
