# The four source datasets — background for slides

Background on the four question sources behind `sampled_questions.json`.
These are well-established, static NLP benchmarks — the facts below don't
change over time, unlike model names/pricing elsewhere in this repo.

## MMLU (Massive Multitask Language Understanding)

**Hendrycks et al., 2021** ("Measuring Massive Multitask Language
Understanding," ICLR 2021). Roughly 14,000–16,000 four-choice multiple-choice
questions spanning 57 subjects — STEM, humanities, social sciences, and
professional fields like law, medicine, and accounting — pitched at
high-school through professional-exam difficulty. Built to test breadth of
factual/academic knowledge picked up during pretraining, not reasoning per
se. Known characteristic: accuracy varies enormously by subject, and later
work has documented a non-trivial rate of label errors/ambiguity in some
subsets.

*Why it matters here:* MMLU questions have a firm, checkable factual anchor
(there's a "right" legal rule or physics fact), which likely explains why it
showed the **lowest** biasing rate in your results — the model has more to
hold onto when a stray suggestion shows up.

## TruthfulQA

**Lin, Hilton & Evans, 2021** ("TruthfulQA: Measuring How Models Mimic
Human Falsehoods," ACL 2022). 817 questions across 38 categories (health,
law, finance, politics, conspiracy theories, common myths). Deliberately
adversarial: every question targets a place where humans commonly believe
something false, so a model trained on human text is tempted to confidently
repeat the popular-but-wrong answer rather than the true one. Comes in both
open-ended and multiple-choice forms. Famous finding from the original
paper: larger/more fluent models were sometimes *less* truthful, because
they'd gotten better at reproducing convincing, common falsehoods.

*Why it matters here:* this dataset is already built to catch a model
leaning on "what sounds right" over "what is right" — which lines up with
it showing one of the **highest** biasing rates in your results. A model
that's already a bit shaky on these questions has less to resist a nudge
with.

## LogiQA

**Liu et al., 2020** ("LogiQA: A Challenge Dataset for Machine Reading
Comprehension with Logical Reasoning," IJCAI 2020). About 8,678 four-choice
questions adapted from the logical-reasoning section of the Chinese Civil
Service Examination, translated into English. Each item is a short passage
plus a question requiring a chain of deductive/inductive/analogical
reasoning — this is testing reasoning process, not factual recall.

*Why it matters here:* a stated wrong belief can derail a reasoning chain
partway through just as easily as it can override a fact — your results
put LogiQA in the middle of the pack, consistent with "some resistance, but
less firm footing than a pure fact."

## HellaSwag

**Zellers et al., 2019** ("HellaSwag: Can a Machine Really Finish Your
Sentence?," ACL 2019). About 70,000 examples (commonly evaluated on a
~10,000-question split), four-choice "pick the most plausible sentence
continuation" format. Built via **adversarial filtering** — wrong answers
were specifically selected because they fool the era's best models on
surface style while being nonsensical on careful reading. Notable finding
since publication: despite being adversarially hard for 2019-era models, it
was "solved" remarkably quickly by later LLMs — a commonly cited example of
how fast benchmarks saturate. Humans score around 95%.

*Why it matters here:* there's no hard fact to fall back on, only a
plausibility judgment — which is likely why it's one of your **highest**
biasing-rate datasets too. "Which ending sounds right" is exactly the kind
of soft, judgment-based question a stated opinion can tip over.

## One-line comparison for a slide

| Dataset | Tests | Size | Anchor type |
|---|---|---|---|
| MMLU | Broad academic knowledge | ~14–16k Qs, 57 subjects | Hard fact |
| TruthfulQA | Resisting popular misconceptions | 817 Qs, 38 categories | Adversarial-to-truth |
| LogiQA | Multi-step logical reasoning | ~8,678 Qs | Reasoning chain |
| HellaSwag | Commonsense plausibility | ~70k Qs (10k test) | Soft judgment |

*(Note: these are general facts about the full public benchmarks. The
subset you're actually testing — 15 questions from each, via the BCT
paper's `suggested_answer` bias split — is a small sample of each, so treat
the "why it matters here" notes as a plausible interpretation of your
results, not a proven mechanism.)*
