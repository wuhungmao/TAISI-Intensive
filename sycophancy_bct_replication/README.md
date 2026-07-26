# Replicating BCT's "suggested answer" sycophancy bias on a newer model

This replicates one specific result from:

> Wei, J. et al. (2024). *Bias-Augmented Consistency Training Reduces
> Biased Reasoning in Chain-of-Thought.* arXiv:2403.05518

The paper studies 9 biases that distort a model's chain-of-thought
reasoning; the one used here is **`suggested_answer`** — a sycophancy bias
where the user states a belief about the answer ("I have this gut feeling
it's A.") before asking a factual multiple-choice question. Prompts and
ground-truth labels come straight from the paper's own released dataset:
[github.com/raybears/cot-transparency](https://github.com/raybears/cot-transparency)
(MIT license), specifically
`dataset_dumps/test/suggested_answer/{mmlu,truthfulqa,logiqa,hellaswag}_suggested_answer.jsonl`.

`sampled_questions.json` in this folder is a pre-drawn sample of 60
questions (15 from each of MMLU, TruthfulQA, LogiQA, HellaSwag, seed=42) —
already extracted into a simpler flat format so you don't need to clone
the ~1GB source repo (which also uses Git LFS for most files, though not
the dataset_dumps files specifically).

## What each question looks like

Every item has:
- `unbiased_prompt` — the question, asked plainly, with instructions to
  think step by step and end with `Therefore, the best answer is: (X).`
- `biased_prompt` — the exact same question, with one line inserted
  stating the user's (often wrong) guess at the answer.
- `ground_truth` — the correct letter.
- `biased_option` — the letter the biased prompt nudges toward. On ~13%
  of sampled items this happens to equal `ground_truth` — a built-in
  control group (the user's stated "guess" is right, so agreeing with it
  isn't sycophancy, it's just being correct).

## Setup (5 min)

Calls go through [OpenRouter](https://openrouter.ai) rather than any single
provider's API directly — one key gets you both Claude and GPT models
through the same OpenAI-compatible client, just by changing the model slug.

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
```

## 1. Run the experiment

```bash
python run_experiment.py
```

With no model specified, this sweeps a default list of 5 models one after
another (see `DEFAULT_MODELS` at the top of `run_experiment.py`):

- `openai/gpt-3.5-turbo` — the exact model the paper used (sanity check)
- `anthropic/claude-sonnet-5`
- `anthropic/claude-opus-4.8`
- `anthropic/claude-haiku-4.5`
- `google/gemini-2.5-pro`

For each model it sends both the unbiased and biased prompt for all 60
questions (120 calls) and saves raw responses to a separate
`results_raw_<model>.json` — so one model's run never overwrites another's.
It's safe to Ctrl-C at any point: re-running skips every model/question/
condition triple you already have, whether that's because it crashed,
you stopped it, or you're adding one more model later.

To run only specific models instead of the default sweep:

```bash
OPENROUTER_MODEL=anthropic/claude-sonnet-5 python run_experiment.py          # just one
OPENROUTER_MODELS=anthropic/claude-sonnet-5,google/gemini-2.5-pro python run_experiment.py  # a custom list
```

Two things worth knowing before you kick off the full sweep:
- **Cost/time**: 5 models × 120 calls = 600 calls total. Opus in particular
  is the most expensive model in the list — if you want a cheaper first
  pass, drop it via `OPENROUTER_MODELS` and add it back later.
- **Model slugs move fast.** These are pinned to specific versions (not
  `-latest` aliases) so the experiment stays reproducible, but double-check
  each one still resolves at [openrouter.ai/models](https://openrouter.ai/models)
  before a real run — if one 404s, `run_experiment.py` will print that and
  skip just that model rather than crash the whole sweep. GPT-3.5-Turbo
  specifically is on OpenAI's own deprecation path for October 2026.

## 2. Grade and chart

```bash
python analyze.py
```

This auto-discovers every `results_raw_*.json` file in the folder, so it
works whether you ran one model or several. Prints:
- Per model: overall accuracy and "biasing rate" (% of answers on the 52
  non-control questions that flip to match the suggested *wrong* answer)
  for the unbiased vs. biased condition.
- The same breakdown split out by source dataset (MMLU / TruthfulQA /
  LogiQA / HellaSwag).
- A flagged list of any response it couldn't parse an answer from.

And saves `bct_replication_results.png` — a chart comparing every model you
ran against the paper's published GPT-3.5-Turbo baseline (hardcoded from
their Table 4 / Table 5: 12.5% → 35.5% biasing rate, 61.7% → 48.0% accuracy,
unbiased → biased). That hardcoded baseline is the one number in this repo
you should double check against the actual paper before presenting — I
pulled it from Table 4/5 via a summarization pass, not a manual read of the
PDF.

## What's actually interesting to say in your presentation

- Does your model's biasing-rate *jump* (unbiased → biased) look smaller
  than GPT-3.5-Turbo's 23-point jump? That's the headline "did a newer/more
  aligned model get more robust to this?" result.
- Does the natural noise floor (biasing rate under the *unbiased* prompt —
  i.e., how often the model happens to pick that option with no nudge at
  all) differ from GPT-3.5's 12.5%? A very low number here matters for how
  you interpret the biased-condition number.
- The per-dataset breakdown might show the effect concentrated in one task
  type (e.g. LogiQA's harder logic questions vs. MMLU trivia) — worth a
  sentence even if you don't have time to dig deeper.
- Be upfront about n=60: single-question flips move the rate by ~2 points
  (1/52 ≈ 1.9%), so treat differences under ~5 points as noise, not signal.

## If you want more than 60 questions

The full dataset has far more available per category (2000 MMLU, 2000
HellaSwag, 817 TruthfulQA, 649 LogiQA) — clone the repo and re-sample if
you want a bigger n:

```bash
git clone https://github.com/raybears/cot-transparency.git
```

(Git LFS is only needed for files under `data/`, not `dataset_dumps/`, so
a plain clone should get you the files you need.)
