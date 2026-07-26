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

Calls within a single model's batch run CONCURRENTLY (a thread pool, since
these are I/O-bound network calls) -- by default 8 at once, tune with
CONCURRENCY. Models themselves are still run one after another, so
per-model progress stays easy to read and each model's results file stays
independent.

Results are written to a PER-MODEL file (results_raw_<model>.json), so
running multiple models never clobbers a previous one -- analyze.py picks
up every results_raw_*.json file it finds automatically. Raw responses are
written to disk as they come in, so Ctrl-C is always safe: re-running
resumes (skips model/question/condition triples you already have), whether
that's because it crashed, you stopped it, or you're adding one more model
later.

Usage:
    export OPENROUTER_API_KEY=sk-or-...

    # Run the full default sweep (paper's model + Claude family + Gemini):
    python run_experiment.py

    # Or just one model:
    OPENROUTER_MODEL=anthropic/claude-sonnet-5 python run_experiment.py

    # Or a custom list:
    OPENROUTER_MODELS=anthropic/claude-sonnet-5,google/gemini-2.5-pro python run_experiment.py

    # Tune concurrency (default 8; lower this if you start seeing a lot of
    # rate-limit retries in the output):
    CONCURRENCY=4 python run_experiment.py
"""

import concurrent.futures
import json
import os
import re
import threading
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
# Some models (Gemini 2.5 Pro, Claude/GPT with extended thinking, etc.) spend
# part of the token budget on a HIDDEN reasoning pass before writing any
# visible text. Turns out some endpoints (Gemini 2.5 Pro among them) treat
# that reasoning as MANDATORY and reject a request that tries to disable it
# outright (400: "Reasoning is mandatory for this endpoint and cannot be
# disabled") -- so instead of turning it off, we just cap it, which is
# accepted whether reasoning is optional or mandatory for a given model.
# Overall max_tokens has to comfortably exceed the reasoning cap, or there's
# nothing left for the visible step-by-step CoT + final answer line.
# See: https://openrouter.ai/docs/use-cases/reasoning-tokens
REASONING_BUDGET = 1000
MAX_TOKENS = 3000  # reasoning budget + plenty of room for the visible answer
TEMPERATURE = 0
CONDITIONS = ["unbiased", "biased"]
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
REASONING_EXTRA_BODY = {"reasoning": {"max_tokens": REASONING_BUDGET}}

# Matches the exact format the dataset's own prompts ask for:
# 'Therefore, the best answer is: (X).'
FINAL_ANSWER_RE = re.compile(r"best answer is:?\s*\(?([A-Za-z])\)?", re.IGNORECASE)


class FatalAuthError(Exception):
    """The API key itself was rejected -- no point retrying anything."""


class ModelNotFoundError(Exception):
    """OpenRouter doesn't recognize this model slug -- skip the rest of this model."""


class BadRequestFatalError(Exception):
    """A malformed/rejected request (400) -- retrying the identical request
    will just fail identically, so don't burn retries on it."""


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
                extra_body=REASONING_EXTRA_BODY,
            )
            content = resp.choices[0].message.content
            return content or ""
        except openai.AuthenticationError as e:
            raise FatalAuthError(str(e))
        except openai.NotFoundError as e:
            raise ModelNotFoundError(str(e))
        except openai.BadRequestError as e:
            raise BadRequestFatalError(str(e))
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError) as e:
            last_err = e
            wait = 2 ** attempt
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


def load_results(path):
    if path.exists():
        return json.loads(path.read_text())
    return []


def run_for_model(client, model, questions):
    path = results_path(model)
    results = load_results(path)
    total = len(questions) * len(CONDITIONS)

    # Drop any previously-saved entry where we never actually got a parseable
    # answer (e.g. a response that got cut off before reaching the answer
    # line) -- treat those as not-done so they're retried instead of skipped
    # forever.
    n_before = len(results)
    results = [r for r in results if r.get("extracted_answer") is not None]
    n_dropped = n_before - len(results)
    if n_dropped:
        print(f"    dropping {n_dropped} previously-unparseable result(s) from {path}, will retry")

    done_keys = {(r["question_id"], r["condition"]) for r in results}

    print(f"\n=== {model}  ->  {path} ===")
    todo = [(q, cond) for q in questions for cond in CONDITIONS if (q["id"], cond) not in done_keys]
    if not todo:
        print(f"    already complete ({total}/{total}), skipping")
        return
    print(f"    {len(todo)} calls to make ({len(done_keys)}/{total} already done), concurrency={CONCURRENCY}")

    write_lock = threading.Lock()
    stop_model = threading.Event()
    completed_count = [len(done_keys)]  # mutable box so the closure below can update it

    def worker(q, condition):
        if stop_model.is_set():
            return None
        prompt = q[f"{condition}_prompt"]
        return call_model(client, model, prompt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_item = {executor.submit(worker, q, cond): (q, cond) for q, cond in todo}
        for future in concurrent.futures.as_completed(future_to_item):
            q, condition = future_to_item[future]
            try:
                raw_text = future.result()
            except FatalAuthError as e:
                stop_model.set()
                raise SystemExit(
                    f"OpenRouter rejected the API key (401 authentication error): {e}\n"
                    f"Double-check OPENROUTER_API_KEY is set to a valid OpenRouter key "
                    f"(starts with 'sk-or-'), not a key from another provider."
                )
            except ModelNotFoundError as e:
                if not stop_model.is_set():
                    print(f"    Model '{model}' not found on OpenRouter (404) -- abandoning "
                          f"remaining calls for it. Check https://openrouter.ai/models for the "
                          f"current slug: {e}")
                stop_model.set()
                continue
            except BadRequestFatalError as e:
                if not stop_model.is_set():
                    print(f"    '{model}' rejected our request shape (400) -- abandoning remaining "
                          f"calls for it rather than retrying an identical failure: {e}")
                stop_model.set()
                continue
            except Exception as e:
                print(f"    FAILED {q['id']}/{condition}, giving up on this one: {e}")
                continue

            if raw_text is None:  # stop_model was already set before this one started
                continue

            final_answer = extract_final_answer(raw_text)
            if final_answer is None:
                print(f"    WARNING: could not find 'best answer is: (X)' in response ({q['id']}/{condition})")
            record = {
                "question_id": q["id"],
                "condition": condition,
                "model": model,
                "raw_response": raw_text,
                "extracted_answer": final_answer,
            }
            with write_lock:
                results.append(record)
                completed_count[0] += 1
                path.write_text(json.dumps(results, indent=2))
                n = completed_count[0]
            print(f"[{n}/{total}] {q['id']} ({q['original_dataset']}) / {condition}")

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
          f"(already-completed question/condition pairs are skipped, so a re-run costs less) "
          f"-- concurrency={CONCURRENCY} per model")

    for model in models:
        run_for_model(client, model, questions)

    print("\nAll models done. Run analyze.py to grade and chart.")


if __name__ == "__main__":
    main()
