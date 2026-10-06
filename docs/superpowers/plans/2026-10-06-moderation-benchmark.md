# Moderation Benchmark (Step 1 of Option A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (chosen by the user) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score off-the-shelf moderation models on one shared Spanish/Portuguese/English exam, so choosing the chat moderator rests on evidence instead of 89 unreviewed generated examples. The models: Laya zero-shot, the app's current Laya question, Detoxify multilingual and an mmBERT toxicity model, all open-source and self-hosted.

**Architecture:** A new `bench/` package, separate from the chat app.
- *Sources* map public datasets and a hand-written Rioplatense casino suite onto one taxonomy.
- *Contestants* return raw model outputs plus a pure mapping onto that taxonomy.
- The *runner* caches raw outputs per batch, keyed by example id and text hash. Runs can resume, reviewed rows are the only new inference, and a mapping fix needs no re-inference.
- The *scoreboard* sets one operating threshold per contestant on a 20% calibration split. It reports per-source AUROC with clustered bootstrap CIs, paired comparisons and per-functionality accuracy in Markdown (Spanish) and JSON.
- The CLI is `scripts/bench.py`.

**Tech Stack:** Python 3.13, uv, pytest, HF `datasets`, `transformers` 5.18 (already locked via laya), `laya==0.3.22`, `detoxify`, `scikit-learn`.

**Spec:** The **Context** and **Design** sections below are the spec. Task 0 copies this file to `docs/superpowers/plans/2026-10-06-moderation-benchmark.md`.

---

## Context

**Situation**
- The team will launch a player chat on the casino/poker platform.
- It needs a moderator for toxicity, hate speech and targeted hate in Spanish (Rioplatense lunfardo), Brazilian Portuguese and English.
- There is **no real chat data yet**.
- The Laya prototype only answers insult/clean plus sentiment. Its 37%→79% F1 came from 89 generated examples whose labels nobody reviewed.

**Research findings**
- Laya is three weeks old, has a single maintainer and no moderation benchmarks, and its CPU cost grows with each question.
- pysentimiento's hate model was trained on non-commercial data.
- Perspective API ends after 2026.

**Agreed direction (option A)**
- Step 1, this plan: a benchmark with a taxonomy, test suites, a harness and a first scoreboard.
- Step 2, its own plan, judged on this benchmark: fine-tune our own multi-label encoder (mmBERT-small or XLM-R, compared against RoBERTuito for Spanish), plus a de-obfuscation normalizer with an editable lexicon.
- Step 3: a cascade with LLM escalation.

**User constraints**
- Only **Spanish** reviewers are available, so only a Spanish suite is drafted. PT and EN rely on public datasets.
- Execution is **subagent-driven**.
- **Open-source and self-hosted only.** Every model needs an open license and must run on our own hardware. No paid or hosted APIs, and no message leaves our infrastructure.

**Outcome**
- `uv run python scripts/bench.py run` writes `bench_out/scoreboard.md` and `.json`.
- The first, preliminary scoreboard is committed under `docs/benchmarks/`.
- The real verdict comes once Spanish reviewers have labeled the suite.

## Design

### Taxonomy (`bench/taxonomy.py`)

| Key | Meaning |
|---|---|
| `insult` | insult, name-calling, mockery or harassment aimed at a person |
| `threat` | threat of violence or harm, including doxxing ("sé dónde vivís") |
| `identity_hate` | attack on people for a PROTECTED trait: race/ethnicity, nationality, religion, gender, sexual orientation, gender identity, disability. Class, body and age insults are `insult`. |
| `sexual_harassment` | unwanted sexual comments, requests or advances aimed at a person |
| `profanity` | swearing or vulgar language. Not harmful by itself; policy decides |
| `target` | `player` / `group` / `none`: who a harmful message attacks |

- Labels are `True`, `False` or `None`. `None` means the source doesn't say, so the label isn't scored.
- **`flag`** (should moderation act?) is derived from FLAG_CATEGORIES = `insult`, `threat`, `identity_hate`.
  - It is True if any of them is True.
  - It is False only if all three are known False.
  - Otherwise it is None.
- **`targets_player`** = `target == "player"`. It is scored only on rows where `flag` is True.
- **SCORED** = the 5 categories + `flag` + `targets_player`.

### Sources

Each source's mapping only sets what that source determines. Its explicit rules for when a row counts as clean are written next to the mapping.

| Name | Lang | Origin | License | Rows |
|---|---|---|---|---|
| `suite-es` | es | `suites/es_casino.csv`, human-reviewed (`author` = generated / human) | own | reviewed rows only |
| `hatecheck-es` / `-pt` / `-en` | es/pt/en | `Paul/hatecheck-spanish`, `Paul/hatecheck-portuguese`, `Paul/hatecheck` | CC BY 4.0 | all, minus cases where annotators disagreed |
| `toldbr` | pt | `mteb/told-br` (21k, 0–3 votes per category) | CC BY-SA 4.0 | 4,000 sampled |
| `olidbr` | pt | `dougtrajano/olid-br`, test split | CC BY 4.0 | all 1,738 |

- **Excluded:** OffendES (gated, and its license is contradictory), HatEval and HateBR (non-commercial), and MHS (cut: it duplicates HateCheck-EN and has easy negatives).
- **Splits:** 20% `calib` (where thresholds are set) and 80% `eval` (what gets reported), chosen by a hash of the cluster. A HateCheck template is one cluster.

### Contestants

| Name | What | Taxonomy mapping |
|---|---|---|
| `laya` | Laya Router configured like the app, with 6 zero-shot `choice` questions | one question per category plus `target` |
| `laya-app` | the app's current `QUESTIONS` | `flag` ← P(insult). This is the incumbent baseline. |
| `detoxify` | `Detoxify("multilingual")` | insult←insult, threat←threat, identity_hate←identity_attack, sexual_harassment←sexual_explicit (approximate), profanity←obscene |
| `horizon-mmbert` | `Horizon-Labs/multilingual-toxicity-base` at a pinned revision, safetensors only. **Unvetted publisher, unknown training data.** | same as detoxify |

- A contestant's `flag` score is `max(insult, threat, identity_hate)` when it covers all three, unless it provides `flag` directly.
- All contestants are Apache-2.0 and run locally. Each one declares its `license`, and a test rejects any license that isn't Apache-2.0 or MIT.

### Metrics

- **Per slice** (contestant × source × label, on the eval split):
  - n, positives, prevalence
  - **AUROC with a 95% clustered bootstrap CI**
  - average precision
  - recall, FPR and precision **at the contestant's operating threshold**: the score at which 5% of calib negatives get flagged, so one threshold transfers across sources.
- **Paired bootstrap** of `flag` AUROC against the best contestant in each source.
- **Per-functionality accuracy** at the operating threshold: `identity_hate` for HateCheck, `flag` for the suite.
- **Too little data:** slices with fewer than 30 rows per class print "pocos datos" instead of numbers.

## Global Constraints

- **Python and tooling:** Python `>=3.11,<3.14` (repo pins 3.13), run through `uv`. Keep `laya==0.3.22` and the locked `transformers` 5.18 / `torch` 2.14 (CUDA index). **Never downgrade them.** If a new dependency needs a downgrade, stop and report BLOCKED.
- **Dependency group:** new deps go in a `bench` dependency group listed in `[tool.uv] default-groups`.
- **Network:** a TLS-inspecting proxy sits in front of this machine. Run every `uv` command with `UV_SYSTEM_CERTS=1` (e.g. `UV_SYSTEM_CERTS=1 uv run pytest`).
- **GPU:** the machine has an NVIDIA RTX 500 Ada Laptop GPU with 4 GB. Only one contestant's model fits at a time, so contestants expose `unload()` and the runner and the latency command call it when a contestant is done.
- **Fast suite** (`uv run pytest`) must never download a model or dataset and must use fakes. Anything that downloads is `@pytest.mark.slow`.
- **Imports:** heavy libraries (`torch`, `transformers`, `datasets`, `detoxify`, `sklearn`) are imported **inside functions**, the same pattern as `app/classifier.py:build_router`.
- **Language:** code, comments and docstrings in English. CLI output, `bench_out/scoreboard.md` and `docs/benchmark.md` in Spanish (repo convention: `scripts/*.py` print in Spanish).
- **Encoding:** always pass `encoding="utf-8"` (or `"utf-8-sig"` for the suite CSV) when reading or writing text. The default codepage on this Windows machine is not UTF-8.
- **App isolation:** don't change the chat app's behavior. The only app change is an optional `device` parameter on `app/classifier.py:build_router`.
- **HF model safety:** never `trust_remote_code=True`. HF models load at a pinned revision SHA.
- **Data licenses:** only own data, CC BY or CC BY-SA. No non-commercial datasets or models.
- **Models:** open-source (Apache-2.0 or MIT) and self-hosted only. No paid or hosted APIs (OpenAI, Azure, Perspective, …), and no message leaves the machine. Every contestant declares its `license`, and a registry test enforces the allowlist.
- **Git:** `bench_out/` is gitignored. Work on branch `bench/step1-benchmark` (Task 0). Commit messages follow the user's `git-commit-helper` format: header `<EMOJI> <TYPE>(<scope>): <imperative summary>` (max 72 chars, no final period; e.g. `✨ New Feature(bench): add moderation taxonomy`, `✅ Adding Tests(bench): …`, `➕ Adding Dependency(bench): …`, `📖 DOC(bench): …`), a blank line, then `-` bullets saying what changed and why, identifiers in backticks. **No trailer lines at all** (no `Co-Authored-By`, no `Signed-off-by`). Commit from Bash with `git commit -F - <<'EOF' … EOF`. The messages in the steps are summaries.

## Review Focus

1. **A suite CSV re-saved by Spanish-locale Excel** (`;` delimiter, BOM, trailing empty rows, `?` cells) must parse exactly like the original. Task 4 tests this.
2. **A reviewer edits a suite row's text after a run.** That row must be predicted again, while untouched rows come from the cache. Task 9 tests this.
3. **A run killed mid-way** (network drop, rate limit, Ctrl+C) must resume from the last finished batch. A truncated last JSONL line must neither crash loading nor corrupt the next append. Task 9 tests this.
4. **Single-class slices, slices with fewer than 30 per class, and contestants lacking a category** must render as "pocos datos" / "n/a" / "—", never as a crash or a fake `0.00`. Tasks 3, 12 and 13 test this.
5. **A public dataset that changes upstream** (new HateCheck functionality, unexpected label value, vote count out of range) must make the loader raise and name the value, not silently map it to clean. Tasks 6 and 7 test this.

## File Structure

```
bench/
  __init__.py
  taxonomy.py              # categories, Example, gold(), has_gold()          (Task 2)
  metrics.py               # AUROC, AP, thresholds, clustered/paired bootstrap (Task 3)
  seed.py                  # builds the first suite draft                      (Task 5)
  sources/
    __init__.py            # Source registry, sample(), split_of(), load_sources()  (Task 8)
    suite.py               # suite CSV parse/write/agreement                    (Task 4)
    hatecheck.py           # HateCheck es/pt/en                                 (Task 6)
    portuguese.py          # ToLD-Br, OLID-BR                                   (Task 7)
  contestants/
    __init__.py            # FACTORIES registry, build_contestants()            (Task 11)
    base.py                # Contestant protocol, effective_scores, map_labels  (Task 9)
    laya.py                # laya, laya-app                                     (Task 10)
    detoxify.py            # detoxify                                           (Task 11)
    hf.py                  # generic HF multi-label classifier, horizon-mmbert  (Task 11)
  cache.py                 # PredictionCache (raw outputs, JSONL)               (Task 9)
  runner.py                # run()                                              (Task 9)
  scoreboard.py            # build_scoreboard()                                 (Task 12)
  report.py                # render() → Markdown                                (Task 13)
  latency.py               # measure()                                          (Task 13)
  cli.py                   # main() for scripts/bench.py                        (Task 13)
scripts/bench.py           # CLI entry point                                    (Task 13)
scripts/seed_suite.py      # writes suites/es_casino.csv                        (Task 5)
suites/es_casino.csv       # the Spanish suite reviewers edit                   (Task 5)
suites/drafts/es_casino_new.jsonl  # new casino drafts                          (Task 5)
docs/benchmark.md          # labeling policy, review guide, how to run/read     (Tasks 5, 14)
docs/benchmarks/2026-10-06-preliminar.md  # first preview scoreboard            (Task 14)
tests/test_bench_*.py      # one file per module; slow ones in tests/test_bench_slow.py
```

---

### Task 0: Branch and plan copy

**Files:**
- Create: `docs/superpowers/plans/2026-10-06-moderation-benchmark.md` (a copy of this plan)

- [ ] **Step 1: Create the branch.**

```bash
git switch -c bench/step1-benchmark
```

- [ ] **Step 2: Copy the plan.**

```bash
mkdir -p docs/superpowers/plans
cp "C:/Users/facundo.zerbino/.claude/plans/enchanted-snuggling-pond.md" docs/superpowers/plans/2026-10-06-moderation-benchmark.md
```

- [ ] **Step 3: Commit.** Summary: "add moderation benchmark plan".

```bash
git add docs/superpowers/plans/2026-10-06-moderation-benchmark.md
git commit
```

---

### Task 1: Dependencies and compatibility spike

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `.gitignore`

**Interfaces:**
- Produces: importable `datasets`, `detoxify` and `sklearn` in the default environment, plus proof that Detoxify and the pinned Horizon revision load on the locked transformers.

- [ ] **Step 1: Add the `bench` group.** In `pyproject.toml`, append to `[dependency-groups]`:

```toml
# Benchmark of moderation models (bench/, scripts/bench.py).
bench = [
    "datasets>=4.0",
    "detoxify>=0.5.2",
    "scikit-learn>=1.9",
]
```

Under the existing `[tool.uv]`, add:

```toml
default-groups = ["dev", "bench"]
```

Then replace the `markers` line in `[tool.pytest.ini_options]` with:

```toml
markers = ["slow: loads real models or downloads datasets; run with `pytest -m slow`"]
```

- [ ] **Step 2: Sync.**

Run: `uv sync`
Expected: resolves and installs. Then check the pins survived:

Run: `uv pip list | grep -iE "^(laya|transformers|torch) "`
Expected: `laya 0.3.22`, `transformers 5.18.0`, `torch 2.14.1+cu130`. If the resolver wants to change any of them, **stop and report BLOCKED** with the resolver output.

- [ ] **Step 3: Smoke-test Detoxify on transformers 5.18.** This downloads a 1.1 GB checkpoint.

```bash
uv run python -c "from detoxify import Detoxify; m = Detoxify('multilingual', device='cpu'); print(sorted(m.predict(['sos un pelotudo', 'buena mano']).keys()))"
```

Expected: `['identity_attack', 'insult', 'obscene', 'severe_toxicity', 'sexual_explicit', 'threat', 'toxicity']`. If it raises, **stop and report BLOCKED** with the traceback. Do not downgrade transformers.

- [ ] **Step 4: Load the pinned Horizon revision safely.** The plan pins `dbf12a9915275078e580707bc2396b07585da31e`, the repo head on 2026-10-06. This downloads ~1.2 GB:

```bash
uv run python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; r='dbf12a9915275078e580707bc2396b07585da31e'; i='Horizon-Labs/multilingual-toxicity-base'; AutoTokenizer.from_pretrained(i, revision=r); m = M.from_pretrained(i, revision=r, use_safetensors=True); print(m.config.problem_type, m.config.id2label)"
```

Expected: `multi_label_classification {0: 'toxicity', 1: 'severe_toxicity', 2: 'obscene', 3: 'threat', 4: 'insult', 5: 'identity_attack', 6: 'sexual_explicit'}`.

- [ ] **Step 5: Ignore benchmark output.** Append to `.gitignore`:

```
# Benchmark predictions cache and reports (scripts/bench.py)
bench_out/
```

- [ ] **Step 6: Run the fast suite.**

Run: `uv run pytest`
Expected: all existing tests pass.

- [ ] **Step 7: Commit.** Summary: "add benchmark dependency group".

```bash
git add pyproject.toml uv.lock .gitignore
git commit
```

---

### Task 2: Taxonomy

**Files:**
- Create: `bench/__init__.py` (empty), `bench/taxonomy.py`
- Test: `tests/test_bench_taxonomy.py`

**Interfaces:**
- Produces:
  - Constants: `LANGS`, `CATEGORIES`, `FLAG_CATEGORIES`, `PROTECTED`, `TARGETS`, `FLAG = "flag"`, `TARGETS_PLAYER = "targets_player"`, `SCORED`, `NOT_FLAGGED`.
  - `Example(id, source, lang, text, labels, target=None, functionality=None, author=None, cluster=None)` with the `.cluster_key` property.
  - `gold(example, label) -> bool | None`
  - `has_gold(example) -> bool`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_taxonomy.py`:

```python
import pytest

from bench.taxonomy import FLAG, TARGETS_PLAYER, Example, gold, has_gold


def ex(labels, target=None, **kwargs):
    return Example(id="t:1", source="t", lang="es", text="hola", labels=labels, target=target, **kwargs)


def test_flag_is_true_when_any_core_category_is_true_even_if_others_are_unknown():
    assert gold(ex({"insult": True}), FLAG) is True


def test_flag_is_false_only_when_insult_threat_and_identity_hate_are_all_known_false():
    assert gold(ex({"insult": False, "threat": False, "identity_hate": False}), FLAG) is False


def test_flag_is_unknown_when_a_core_category_is_unknown_and_none_is_true():
    assert gold(ex({"insult": False, "identity_hate": False}), FLAG) is None


def test_profanity_and_sexual_harassment_do_not_make_a_message_flagged():
    example = ex({"insult": False, "threat": False, "identity_hate": False, "profanity": True, "sexual_harassment": True})

    assert gold(example, FLAG) is False
    assert gold(example, "sexual_harassment") is True


def test_targets_player_is_only_scored_on_flagged_messages():
    assert gold(ex({"insult": True}, target="player"), TARGETS_PLAYER) is True
    assert gold(ex({"identity_hate": True}, target="group"), TARGETS_PLAYER) is False
    assert gold(ex({"insult": False, "threat": False, "identity_hate": False}, target="none"), TARGETS_PLAYER) is None
    assert gold(ex({"insult": True}), TARGETS_PLAYER) is None


def test_a_category_the_source_does_not_annotate_is_unknown():
    assert gold(ex({}), "threat") is None
    assert has_gold(ex({})) is False
    assert has_gold(ex({"profanity": False})) is True


@pytest.mark.parametrize(
    "override",
    [
        {"lang": "fr"},
        {"labels": {"spam": True}},
        {"labels": {"insult": 1}},
        {"target": "everyone"},
        {"text": "   "},
    ],
)
def test_example_rejects_invalid_fields(override):
    fields = {"id": "t:1", "source": "t", "lang": "es", "text": "hola", "labels": {}}

    with pytest.raises(ValueError):
        Example(**{**fields, **override})


def test_cluster_defaults_to_the_example_id():
    assert ex({}).cluster_key == "t:1"
    assert ex({}, cluster="tmpl-3").cluster_key == "tmpl-3"
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench'`.

- [ ] **Step 3: Implement.** Create an empty `bench/__init__.py` and `bench/taxonomy.py`:

```python
"""The moderation taxonomy every benchmark source and contestant is mapped onto.

docs/benchmark.md has the labeling policy behind each category.
"""

from __future__ import annotations

from dataclasses import dataclass

LANGS = ("es", "pt", "en")
CATEGORIES = ("insult", "threat", "identity_hate", "sexual_harassment", "profanity")
# What `flag` (should moderation act?) is built from. Profanity alone is not harmful: the chat policy
# decides what to do with it. Sexual harassment is scored on its own: only the Spanish suite labels it,
# and the off-the-shelf "sexual" outputs detect sexual content, not harassment.
FLAG_CATEGORIES = ("insult", "threat", "identity_hate")
# Traits whose attack counts as identity_hate. Class, body and age insults are `insult`.
PROTECTED = ("race_ethnicity", "nationality", "religion", "gender", "sexual_orientation", "gender_identity", "disability")
TARGETS = ("player", "group", "none")
FLAG = "flag"
TARGETS_PLAYER = "targets_player"
SCORED = CATEGORIES + (FLAG, TARGETS_PLAYER)
NOT_FLAGGED = {category: False for category in FLAG_CATEGORIES}


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    lang: str
    text: str
    # True / False per category; None or absent when the source does not say, so it is not scored.
    labels: dict
    # Who a harmful message attacks; None when the source does not say.
    target: str | None = None
    functionality: str | None = None
    author: str | None = None
    # Examples generated from one template share a cluster: bootstrap and splits keep them together.
    cluster: str | None = None

    def __post_init__(self):
        if self.lang not in LANGS:
            raise ValueError(f"{self.id}: unknown language {self.lang!r}")
        unknown = sorted(set(self.labels) - set(CATEGORIES))
        if unknown:
            raise ValueError(f"{self.id}: unknown categories {unknown}")
        bad = {k: v for k, v in self.labels.items() if not (v is None or isinstance(v, bool))}
        if bad:
            raise ValueError(f"{self.id}: labels must be True, False or None, got {bad}")
        if self.target is not None and self.target not in TARGETS:
            raise ValueError(f"{self.id}: unknown target {self.target!r}")
        if not self.text.strip():
            raise ValueError(f"{self.id}: empty text")

    @property
    def cluster_key(self) -> str:
        return self.cluster or self.id


def gold(example: Example, label: str) -> bool | None:
    """The correct answer for one scored label, or None when the example cannot score it."""
    if label == FLAG:
        values = [example.labels.get(c) for c in FLAG_CATEGORIES]
        if any(v is True for v in values):
            return True
        if all(v is False for v in values):
            return False
        return None
    if label == TARGETS_PLAYER:
        # On clean messages the target would only repeat `flag`, so it is scored on harmful ones only.
        if gold(example, FLAG) is not True or example.target is None:
            return None
        return example.target == "player"
    return example.labels.get(label)


def has_gold(example: Example) -> bool:
    return any(gold(example, label) is not None for label in SCORED)
```

- [ ] **Step 4: Run the tests again.**

Run: `uv run pytest tests/test_bench_taxonomy.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.** Summary: "add moderation taxonomy".

```bash
git add bench/__init__.py bench/taxonomy.py tests/test_bench_taxonomy.py
git commit
```

---

### Task 3: Metrics

**Files:**
- Create: `bench/metrics.py`
- Test: `tests/test_bench_metrics.py`

**Interfaces:**
- Produces:
  - `MAX_FPR = 0.05`
  - `auroc(gold, scores) -> float | None`
  - `average_precision(gold, scores) -> float | None`
  - `threshold_at_fpr(gold, scores, max_fpr=MAX_FPR) -> float | None`
  - `at_threshold(gold, scores, threshold) -> {"recall", "fpr", "precision"}` (values may be None)
  - `_resample(clusters, rng) -> list[int]`
  - `bootstrap_auroc_ci(gold, scores, clusters, n_boot=1000, seed=0) -> tuple[float, float] | None`
  - `paired_auroc_diff(gold, a, b, clusters, n_boot=1000, seed=0) -> {"diff", "lo", "hi"} | None`
  - `slice_metrics(gold, scores, clusters, threshold, n_boot=1000, seed=0) -> dict` with keys `n, positives, prevalence, auroc, auroc_ci, ap, threshold, recall_at_op, fpr_at_op, precision_at_op`
- In all of these, `gold: list[bool]`, `scores: list[float]`, `clusters: list[str]`.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_metrics.py`:

```python
import random

from bench.metrics import (
    _resample,
    at_threshold,
    auroc,
    average_precision,
    bootstrap_auroc_ci,
    paired_auroc_diff,
    slice_metrics,
    threshold_at_fpr,
)


def test_auroc_is_one_for_perfect_separation_and_none_with_a_single_class():
    assert auroc([True, False, True, False], [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert auroc([True, False], [0.1, 0.9]) == 0.0
    assert auroc([True, True], [0.1, 0.9]) is None
    assert average_precision([False, False], [0.1, 0.2]) is None


def test_threshold_lets_through_at_most_the_allowed_share_of_negatives():
    gold = [False] * 100 + [True] * 10
    scores = [i / 100 for i in range(100)] + [0.99] * 10

    threshold = threshold_at_fpr(gold, scores, max_fpr=0.05)

    assert threshold == 0.94
    assert at_threshold(gold, scores, threshold) == {"recall": 1.0, "fpr": 0.05, "precision": 10 / 15}


def test_with_few_negatives_the_threshold_is_the_highest_negative():
    assert threshold_at_fpr([False, False, True], [0.2, 0.4, 0.9], max_fpr=0.05) == 0.4


def test_threshold_is_none_without_negatives():
    assert threshold_at_fpr([True], [0.5]) is None


def test_only_scores_strictly_above_the_threshold_are_flagged():
    assert at_threshold([True, False], [0.5, 0.5], 0.5) == {"recall": 0.0, "fpr": 0.0, "precision": None}


def test_resample_keeps_clusters_whole():
    clusters = ["a", "a", "b", "c", "c", "c"]

    drawn = _resample(clusters, random.Random(1))

    for cluster in "abc":
        members = [i for i, c in enumerate(clusters) if c == cluster]
        assert len({drawn.count(i) for i in members}) == 1


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible():
    rng = random.Random(0)
    gold = [i % 2 == 0 for i in range(200)]
    scores = [(0.6 if g else 0.4) + rng.uniform(-0.3, 0.3) for g in gold]
    clusters = [str(i) for i in range(200)]

    lo, hi = bootstrap_auroc_ci(gold, scores, clusters, n_boot=200, seed=3)

    assert lo <= auroc(gold, scores) <= hi
    assert bootstrap_auroc_ci(gold, scores, clusters, n_boot=200, seed=3) == (lo, hi)


def test_paired_difference_of_a_model_with_itself_is_zero():
    gold = [True, False] * 50
    scores = [0.7 if g else 0.3 for g in gold]

    assert paired_auroc_diff(gold, scores, scores, [str(i) for i in range(100)], n_boot=50) == {
        "diff": 0.0, "lo": 0.0, "hi": 0.0,
    }


def test_slice_metrics_reports_none_for_a_single_class_or_no_threshold():
    metrics = slice_metrics([True, True], [0.2, 0.9], ["1", "2"], threshold=None, n_boot=10)

    assert metrics["n"] == 2 and metrics["positives"] == 2 and metrics["prevalence"] == 1.0
    assert metrics["auroc"] is None and metrics["auroc_ci"] is None and metrics["ap"] is None
    assert metrics["recall_at_op"] is None and metrics["fpr_at_op"] is None
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.metrics'`.

- [ ] **Step 3: Implement.** Create `bench/metrics.py`:

```python
"""Metrics for one slice of the scoreboard: gold booleans and one contestant's scores for them.

Scores from different contestants live on different scales and most are not calibrated probabilities,
so nothing here assumes 0.5 means anything. Threshold-free AUROC ranks contestants. A per-contestant
operating threshold, chosen where 5% of the calibration negatives get flagged, shows what one fixed
threshold would do on each source.
"""

from __future__ import annotations

import random

MAX_FPR = 0.05


def _both_classes(gold: list[bool]) -> bool:
    return 0 < sum(gold) < len(gold)


def auroc(gold: list[bool], scores: list[float]) -> float | None:
    if not _both_classes(gold):
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(gold, scores))


def average_precision(gold: list[bool], scores: list[float]) -> float | None:
    if not _both_classes(gold):
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(gold, scores))


def threshold_at_fpr(gold: list[bool], scores: list[float], max_fpr: float = MAX_FPR) -> float | None:
    """The score above which at most `max_fpr` of the negatives fall. Flag when score > threshold."""
    negatives = sorted(s for g, s in zip(gold, scores) if not g)
    if not negatives:
        return None
    allowed = int(max_fpr * len(negatives) + 1e-9)
    return negatives[len(negatives) - 1 - allowed]


def at_threshold(gold: list[bool], scores: list[float], threshold: float) -> dict:
    flagged = [s > threshold for s in scores]
    tp = sum(f and g for f, g in zip(flagged, gold))
    fp = sum(f and not g for f, g in zip(flagged, gold))
    positives = sum(gold)
    negatives = len(gold) - positives
    return {
        "recall": tp / positives if positives else None,
        "fpr": fp / negatives if negatives else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def _resample(clusters: list[str], rng: random.Random) -> list[int]:
    """Indices of a bootstrap sample that draws whole clusters with replacement."""
    members: dict[str, list[int]] = {}
    for i, cluster in enumerate(clusters):
        members.setdefault(cluster, []).append(i)
    keys = list(members)
    return [i for _ in keys for i in members[keys[rng.randrange(len(keys))]]]


def _interval(values: list[float], alpha: float = 0.05) -> tuple[float, float] | None:
    if not values:
        return None
    values = sorted(values)
    lo = values[int(alpha / 2 * len(values))]
    hi = values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]
    return lo, hi


def bootstrap_auroc_ci(gold, scores, clusters, n_boot: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    if auroc(gold, scores) is None:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        idx = _resample(clusters, rng)
        value = auroc([gold[i] for i in idx], [scores[i] for i in idx])
        if value is not None:
            values.append(value)
    return _interval(values)


def paired_auroc_diff(gold, a, b, clusters, n_boot: int = 1000, seed: int = 0) -> dict | None:
    """AUROC(a) - AUROC(b) on the same items. Both are scored on each resample, so item difficulty
    cancels out; comparing two separate intervals would not."""
    base_a, base_b = auroc(gold, a), auroc(gold, b)
    if base_a is None or base_b is None:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample(clusters, rng)
        g = [gold[i] for i in idx]
        va, vb = auroc(g, [a[i] for i in idx]), auroc(g, [b[i] for i in idx])
        if va is not None and vb is not None:
            diffs.append(va - vb)
    interval = _interval(diffs)
    return {"diff": base_a - base_b, "lo": interval[0] if interval else None, "hi": interval[1] if interval else None}


def slice_metrics(gold, scores, clusters, threshold: float | None, n_boot: int = 1000, seed: int = 0) -> dict:
    positives = sum(gold)
    n = len(gold)
    op = at_threshold(gold, scores, threshold) if threshold is not None else {"recall": None, "fpr": None, "precision": None}
    return {
        "n": n,
        "positives": positives,
        "prevalence": positives / n if n else None,
        "auroc": auroc(gold, scores),
        "auroc_ci": bootstrap_auroc_ci(gold, scores, clusters, n_boot=n_boot, seed=seed),
        "ap": average_precision(gold, scores),
        "threshold": threshold,
        "recall_at_op": op["recall"],
        "fpr_at_op": op["fpr"],
        "precision_at_op": op["precision"],
    }
```

- [ ] **Step 4: Run the tests again.**

Run: `uv run pytest tests/test_bench_metrics.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.** Summary: "add benchmark metrics with clustered and paired bootstrap".

```bash
git add bench/metrics.py tests/test_bench_metrics.py
git commit
```

---

### Task 4: Suite CSV format

**Files:**
- Create: `bench/sources/__init__.py` (temporarily empty; Task 8 fills it), `bench/sources/suite.py`
- Test: `tests/test_bench_suite.py`

**Interfaces:**
- Consumes: `bench.taxonomy.Example`, `CATEGORIES`, `LANGS`, `TARGETS`.
- Produces:
  - `COLUMNS` (13 names, in order)
  - `AUTHORS = ("generated", "human")`
  - `SuiteError(ValueError)`
  - `parse_suite(text, source) -> list[tuple[Example, bool]]` (each example plus its reviewed flag)
  - `load_suite(path, source, include_unreviewed=False) -> list[Example]`
  - `write_suite(path, rows: list[dict[str, str]]) -> None`
  - `agreement(a: list[Example], b: list[Example]) -> dict[category, {"n", "kappa"}]`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_suite.py`:

```python
import pytest

from bench.sources.suite import COLUMNS, SuiteError, agreement, load_suite, parse_suite, write_suite

ROW = ["es-0001", "es", "insult_direct", "generated", "sos un manco", "1", "0", "0", "", "?", "player", "1", ""]


def csv_text(*rows, delimiter=";"):
    return "\n".join([delimiter.join(COLUMNS), *(delimiter.join(r) for r in rows)]) + "\n"


def with_cell(column, value, row=ROW):
    changed = list(row)
    changed[COLUMNS.index(column)] = value
    return changed


def test_parses_a_reviewed_row():
    [(example, reviewed)] = parse_suite(csv_text(ROW), "suite-es")

    assert example.id == "suite-es:es-0001"
    assert example.labels == {
        "insult": True, "threat": False, "identity_hate": False, "sexual_harassment": None, "profanity": None,
    }
    assert (example.target, example.author, example.functionality) == ("player", "generated", "insult_direct")
    assert reviewed is True


def test_excel_semicolons_with_bom_parse_like_plain_commas():
    assert parse_suite("\ufeff" + csv_text(ROW), "s") == parse_suite(csv_text(ROW, delimiter=","), "s")


def test_empty_rows_excel_leaves_at_the_end_are_ignored():
    text = csv_text(ROW) + ";" * (len(COLUMNS) - 1) + "\n\n"

    assert len(parse_suite(text, "s")) == 1


def test_bad_cell_names_its_line_and_column():
    with pytest.raises(SuiteError, match=r"línea 2.*insult"):
        parse_suite(csv_text(with_cell("insult", "si")), "s")


def test_duplicate_ids_are_rejected():
    with pytest.raises(SuiteError, match="repetido"):
        parse_suite(csv_text(ROW, ROW), "s")


@pytest.mark.parametrize("column,value", [("lang", "fr"), ("author", "bot"), ("target", "todos"), ("reviewed", "sí"), ("text", "")])
def test_invalid_values_are_rejected_with_the_line(column, value):
    with pytest.raises(SuiteError, match="línea 2"):
        parse_suite(csv_text(with_cell(column, value)), "s")


def test_missing_column_is_reported():
    with pytest.raises(SuiteError, match="faltan columnas"):
        parse_suite("id;text\nes-1;hola\n", "s")


def test_load_keeps_only_reviewed_rows_unless_asked(tmp_path):
    path = tmp_path / "suite.csv"
    path.write_text(csv_text(ROW, with_cell("reviewed", "0", with_cell("id", "es-0002"))), encoding="utf-8")

    assert [e.id for e in load_suite(path, "s")] == ["s:es-0001"]
    assert len(load_suite(path, "s", include_unreviewed=True)) == 2


def test_written_suite_round_trips_with_bom_semicolons_and_awkward_text(tmp_path):
    path = tmp_path / "suite.csv"
    row = dict(zip(COLUMNS, ROW)) | {"text": 'dale; vení,\n"ya"'}

    write_suite(path, [row])

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8-sig").splitlines()[0] == ";".join(COLUMNS)
    [example] = load_suite(path, "s")
    assert example.text == 'dale; vení,\n"ya"'


def test_agreement_is_cohens_kappa_per_category():
    rows_a = [with_cell("insult", v, with_cell("id", f"es-{i}")) for i, v in enumerate(["1", "0", "1", "0"])]
    rows_b = [with_cell("insult", v, with_cell("id", f"es-{i}")) for i, v in enumerate(["1", "0", "0", "0"])]
    a = [e for e, _ in parse_suite(csv_text(*rows_a), "s")]
    b = [e for e, _ in parse_suite(csv_text(*rows_b), "s")]

    assert agreement(a, a)["insult"] == {"n": 4, "kappa": 1.0}
    assert agreement(a, b)["insult"] == {"n": 4, "kappa": 0.5}
    assert agreement(a, b)["profanity"] == {"n": 0, "kappa": None}
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_suite.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.sources'`.

- [ ] **Step 3: Implement.** Create an empty `bench/sources/__init__.py` and `bench/sources/suite.py`:

```python
"""The hand-written suite: a CSV that reviewers edit in Excel or Google Sheets.

Category cells hold 1 (yes), 0 (no), ? (reviewers disagree or can't tell) or nothing (not reviewed).
? and empty are both left out of scoring. Spanish-locale Excel saves with `;` and a BOM, so both
delimiters are read; the file is written with `;` so a double click opens it in columns.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from bench.taxonomy import CATEGORIES, Example

COLUMNS = ("id", "lang", "functionality", "author", "text", *CATEGORIES, "target", "reviewed", "notes")
AUTHORS = ("generated", "human")


class SuiteError(ValueError):
    pass


def _cell(value: str, line: int, column: str) -> bool | None:
    if value == "1":
        return True
    if value == "0":
        return False
    if value in ("", "?"):
        return None
    raise SuiteError(f"línea {line}, columna {column}: {value!r} no es 1, 0, ? ni vacío")


def parse_suite(text: str, source: str) -> list[tuple[Example, bool]]:
    """(example, reviewed) for every non-empty row. Raises SuiteError naming the line of the first problem."""
    text = text.lstrip("\ufeff")
    header = text.split("\n", 1)[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise SuiteError(f"faltan columnas: {', '.join(missing)}")

    parsed, seen = [], set()
    for raw in reader:
        row = {c: (raw.get(c) or "").strip() for c in COLUMNS}
        if not any(row.values()):
            continue
        line = reader.line_num
        if not row["id"] or row["id"] in seen:
            raise SuiteError(f"línea {line}: id vacío o repetido ({row['id']!r})")
        seen.add(row["id"])
        if row["author"] not in AUTHORS:
            raise SuiteError(f"línea {line}: author tiene que ser {' o '.join(AUTHORS)}, no {row['author']!r}")
        if row["reviewed"] not in ("1", "0", ""):
            raise SuiteError(f"línea {line}: reviewed tiene que ser 1 o 0, no {row['reviewed']!r}")
        labels = {c: _cell(row[c], line, c) for c in CATEGORIES}
        try:
            example = Example(
                id=f"{source}:{row['id']}",
                source=source,
                lang=row["lang"],
                text=row["text"],
                labels=labels,
                target=row["target"] or None,
                functionality=row["functionality"] or None,
                author=row["author"],
            )
        except ValueError as e:
            raise SuiteError(f"línea {line}: {e}") from None
        parsed.append((example, row["reviewed"] == "1"))
    return parsed


def load_suite(path: Path, source: str, include_unreviewed: bool = False) -> list[Example]:
    rows = parse_suite(Path(path).read_text(encoding="utf-8-sig"), source)
    return [example for example, reviewed in rows if reviewed or include_unreviewed]


def write_suite(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, delimiter=";")
    writer.writeheader()
    writer.writerows({c: row.get(c, "") for c in COLUMNS} for row in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8-sig", newline="")


def agreement(a: list[Example], b: list[Example]) -> dict[str, dict]:
    """Cohen's kappa per category between two reviewers, on the rows both labeled 1 or 0."""
    from sklearn.metrics import cohen_kappa_score

    other = {e.id: e for e in b}
    result = {}
    for category in CATEGORIES:
        pairs = [(e.labels.get(category), other[e.id].labels.get(category)) for e in a if e.id in other]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
        # With a single class in both columns kappa is undefined (sklearn returns nan).
        kappa = float(cohen_kappa_score(xs, ys)) if len(set(xs) | set(ys)) > 1 else None
        result[category] = {"n": len(pairs), "kappa": kappa}
    return result
```

- [ ] **Step 4: Run the tests again.**

Run: `uv run pytest tests/test_bench_suite.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.** Summary: "add suite CSV format for reviewers".

```bash
git add bench/sources/__init__.py bench/sources/suite.py tests/test_bench_suite.py
git commit
```

### Task 5: Spanish suite draft and labeling policy

**Files:**
- Create: `bench/seed.py`, `scripts/seed_suite.py`, `suites/drafts/es_casino_new.jsonl`, `suites/es_casino.csv` (generated by the script), `docs/benchmark.md`
- Test: `tests/test_bench_seed.py`

**Interfaces:**
- Consumes:
  - `bench.sources.suite.COLUMNS`, `write_suite`, `load_suite`
  - `bench.taxonomy.CATEGORIES`, `TARGETS`
  - `app.examples.normalize_text`
- Produces:
  - `SEED_MAP`
  - `seed_rows(generated) -> list[dict[str, str]]`
  - `draft_rows(drafts) -> list[dict[str, str]]`
  - `build_suite(generated, drafts) -> list[dict[str, str]]`
  - `write_new_suite(path, rows, force=False)`
  - `read_jsonl(path) -> list[dict]`
  - the committed `suites/es_casino.csv` (651 rows, all `reviewed=0`)

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_seed.py`:

```python
from collections import Counter
from pathlib import Path

import pytest

from bench.seed import build_suite, draft_rows, seed_rows, write_new_suite
from bench.sources.suite import load_suite, parse_suite

ROOT = Path(__file__).resolve().parent.parent
DRAFT_COUNTS = {
    "gaming_insult": 25, "bets_amounts": 30, "obfuscated_insult": 25, "identity_hate_casino": 25,
    "obfuscated_slur": 15, "affectionate_identity_term": 10, "sexual_harassment": 20, "threat_doxxing": 15,
    "quoting_reporting_casino": 10, "self_deprecation": 10, "profanity_at_game": 15, "code_mixing_insult": 10,
    "code_mixing_banter": 10, "counter_speech": 10,
}


def test_seed_sets_only_what_the_category_implies():
    [row] = seed_rows([{"text": "callate gil", "label": "insult", "category": "insulto directo"}])

    assert (row["functionality"], row["insult"], row["target"]) == ("insult_direct", "1", "player")
    assert row["threat"] == row["identity_hate"] == row["sexual_harassment"] == row["profanity"] == ""
    assert (row["lang"], row["author"], row["reviewed"]) == ("es", "generated", "0")


def test_everyday_messages_are_seeded_clean_in_every_category():
    [row] = seed_rows([{"text": "mañana a las 10", "label": "clean", "category": "cotidiano"}])

    assert [row[c] for c in ("insult", "threat", "identity_hate", "sexual_harassment", "profanity")] == ["0"] * 5


def test_unknown_seed_category_is_an_error():
    with pytest.raises(ValueError, match="categoría"):
        seed_rows([{"text": "x", "label": "clean", "category": "spam"}])


def test_drafts_keep_their_labels_and_reject_bad_targets():
    [row] = draft_rows([{"functionality": "gaming_insult", "text": "sos un manco", "insult": 1, "target": "player"}])

    assert (row["insult"], row["threat"], row["target"]) == ("1", "", "player")
    with pytest.raises(ValueError):
        draft_rows([{"functionality": "x", "text": "y", "target": "todos"}])


def test_build_assigns_sequential_ids_and_rejects_duplicate_texts():
    rows = build_suite([{"text": "Sos un gil", "label": "insult", "category": "insulto directo"}],
                       [{"functionality": "gaming_insult", "text": "manco", "insult": "1", "target": "player"}])

    assert [r["id"] for r in rows] == ["es-0001", "es-0002"]
    with pytest.raises(ValueError, match="repetido"):
        build_suite([{"text": "Sos un gil", "label": "insult", "category": "insulto directo"}],
                    [{"functionality": "gaming_insult", "text": "sos  un GIL", "insult": "1"}])


def test_refuses_to_overwrite_a_suite_that_may_hold_reviews(tmp_path):
    path = tmp_path / "suite.csv"
    rows = build_suite([{"text": "hola", "label": "clean", "category": "cotidiano"}], [])
    write_new_suite(path, rows)

    with pytest.raises(FileExistsError):
        write_new_suite(path, rows)
    write_new_suite(path, rows, force=True)
    assert [e.text for e, _ in parse_suite(path.read_text(encoding="utf-8-sig"), "s")] == ["hola"]


def test_committed_suite_has_the_seed_and_every_draft_functionality():
    examples = load_suite(ROOT / "suites" / "es_casino.csv", "suite-es", include_unreviewed=True)
    counts = Counter(e.functionality for e in examples)

    assert len(examples) == 421 + sum(DRAFT_COUNTS.values())
    for functionality, n in DRAFT_COUNTS.items():
        assert counts[functionality] == n, functionality
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.seed'`.

- [ ] **Step 3: Implement the seeding.** Create `bench/seed.py`:

```python
"""Builds the first draft of suites/es_casino.csv: the 421 generated Rioplatense examples plus new
casino drafts.

A generated example only gets the labels its category implies; every other cell stays empty for the
reviewers. The suggestions come from a generator, so they count only after a person reviews them.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.examples import normalize_text
from bench.sources.suite import COLUMNS, write_suite
from bench.taxonomy import CATEGORIES, TARGETS

CLEAN = {"insult": "0", "threat": "0", "identity_hate": "0"}
# category in datasets/generated_rioplatense_v1.jsonl → (functionality, implied labels, target)
SEED_MAP = {
    "insulto directo": ("insult_direct", {"insult": "1"}, "player"),
    "amenaza": ("threat", {"threat": "1"}, "player"),
    # Several of these are class, body or age insults ("villero", "gorda", "viejo"): reviewers move them
    # to insult. Target is left for them too, since many are aimed at the person.
    "discriminación": ("identity_hate", {"identity_hate": "1"}, ""),
    "puteada sin destinatario": ("profanity_untargeted", {**CLEAN, "profanity": "1"}, "none"),
    "boludo amistoso": ("banter_friendly", CLEAN, "none"),
    "negación": ("negation", CLEAN, "none"),
    "sarcasmo": ("sarcasm", CLEAN, "none"),
    "hablar de insultos": ("quoting_reporting", CLEAN, "none"),
    "crítica sin insulto": ("criticism_no_insult", CLEAN, "none"),
    "cotidiano": ("everyday", {**CLEAN, "sexual_harassment": "0", "profanity": "0"}, "none"),
}


def _row(functionality: str, text: str, labels: dict, target: str, notes: str = "") -> dict[str, str]:
    if target not in TARGETS + ("",):
        raise ValueError(f"target desconocido {target!r} en {text!r}")
    row = dict.fromkeys(COLUMNS, "")
    row.update(lang="es", functionality=functionality, author="generated", text=text.strip(), target=target,
               reviewed="0", notes=notes)
    for category, value in labels.items():
        value = str(value)
        if category not in CATEGORIES or value not in ("1", "0", "?", ""):
            raise ValueError(f"etiqueta inválida {category}={value!r} en {text!r}")
        row[category] = value
    return row


def seed_rows(generated: list[dict]) -> list[dict[str, str]]:
    rows = []
    for example in generated:
        if example["category"] not in SEED_MAP:
            raise ValueError(f"categoría sin mapeo: {example['category']!r}")
        functionality, labels, target = SEED_MAP[example["category"]]
        rows.append(_row(functionality, example["text"], labels, target))
    return rows


def draft_rows(drafts: list[dict]) -> list[dict[str, str]]:
    return [
        _row(d["functionality"], d["text"], {c: d.get(c, "") for c in CATEGORIES}, d.get("target", ""), d.get("notes", ""))
        for d in drafts
    ]


def build_suite(generated: list[dict], drafts: list[dict]) -> list[dict[str, str]]:
    rows = seed_rows(generated) + draft_rows(drafts)
    seen = set()
    for i, row in enumerate(rows, start=1):
        key = normalize_text(row["text"])
        if key in seen:
            raise ValueError(f"texto repetido: {row['text']!r}")
        seen.add(key)
        row["id"] = f"es-{i:04d}"
    return rows


def write_new_suite(path: Path, rows: list[dict[str, str]], force: bool = False) -> None:
    if Path(path).exists() and not force:
        raise FileExistsError(f"{path} ya existe y puede tener revisiones. Usá --force para pisarlo.")
    write_suite(path, rows)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
```

Create `scripts/seed_suite.py`:

```python
"""Write the first draft of suites/es_casino.csv for review.

    uv run python scripts/seed_suite.py           # refuses to overwrite a suite that may hold reviews
    uv run python scripts/seed_suite.py --force
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.seed import build_suite, read_jsonl, write_new_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="pisar la suite aunque ya exista")
    args = parser.parse_args()

    rows = build_suite(
        read_jsonl(ROOT / "datasets" / "generated_rioplatense_v1.jsonl"),
        read_jsonl(ROOT / "suites" / "drafts" / "es_casino_new.jsonl"),
    )
    try:
        write_new_suite(ROOT / "suites" / "es_casino.csv", rows, force=args.force)
    except FileExistsError as e:
        print(e)
        return 1
    print(f"Suite escrita con {len(rows)} filas, todas pendientes de revisión:")
    for functionality, n in sorted(Counter(r["functionality"] for r in rows).items()):
        print(f"  {functionality:<28} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Draft the new casino rows.** Write `suites/drafts/es_casino_new.jsonl`, one JSON object per line with these keys:
  - `functionality`, `text`
  - any of `insult`, `threat`, `identity_hate`, `sexual_harassment`, `profanity` as `"1"`, `"0"` or `"?"`; omitted means empty
  - `target`
  - optional `notes`

The **exact count** per functionality below is enforced by the test in Step 1. Columns not listed stay empty.

| functionality | rows | labels | target | examples (write your own; don't copy these) |
|---|---|---|---|---|
| `gaming_insult` | 25 | insult=1 | player | "sos un manco, desinstalá" · "rata, siempre te vas al mazo" |
| `bets_amounts` | 30 | insult=0 threat=0 identity_hate=0 sexual_harassment=0 profanity=0 | none | "all in 500" · "pago x2 la próxima" · "gg wp" · "te debo 1k" · "p4r de 4s" |
| `obfuscated_insult` | 25 | insult=1 | player | "sos un p3lotud0" · "p e l o t u d o" · "forrooooo" |
| `identity_hate_casino` | 25 | identity_hate=1 | player or group as written | "bolita de mierda volvete a tu país" · "los trolos no deberían jugar" |
| `obfuscated_slur` | 15 | identity_hate=1 | player or group | "tr0l0" · "b0l1t4 de m13rd4" |
| `affectionate_identity_term` | 10 | insult=0 threat=0 identity_hate=0 | none | "negro, arrancamos?" · "gordo, qué manito" |
| `sexual_harassment` | 20 | sexual_harassment=1 | player | "mandame fotos y te paso fichas" · "pasame tu número, linda" |
| `threat_doxxing` | 15 | threat=1 | player | "sé dónde vivís" · "te espero a la salida del casino" |
| `quoting_reporting_casino` | 10 | insult=0 threat=0 identity_hate=0 | none | "me dijo trolo, ya lo reporté" |
| `self_deprecation` | 10 | insult=0 threat=0 identity_hate=0 | none | "soy un manco, perdí todo" |
| `profanity_at_game` | 15 | insult=0 threat=0 identity_hate=0 profanity=1 | none | "qué mano de mierda" · "la puta madre, otra vez el river" |
| `code_mixing_insult` | 10 | insult=1 | player | "nice hand manco" · "ez noob boludo" |
| `code_mixing_banter` | 10 | insult=0 threat=0 identity_hate=0 | none | "gg boludo" · "nh genio" |
| `counter_speech` | 10 | identity_hate=0 | none | "no digas bolita, es racista" |

Drafting rules:
- **Style:** Rioplatense voseo and real chat style: mostly under 60 characters, often lowercase, little punctuation, varied wording.
- **Use the real terms.** This is a moderation test suite, so it needs real slurs and insults where the functionality calls for them.
- **No real full names.** Common first names only.
- **No duplicates.** Never repeat or near-repeat a text from `datasets/generated_rioplatense_v1.jsonl`; `build_suite` rejects normalized duplicates.
- **Notes:** when a row is deliberately borderline, say why in `notes` so reviewers see the intent.

- [ ] **Step 5: Generate the suite.**

Run: `uv run python scripts/seed_suite.py`
Expected: `Suite escrita con 651 filas…` and a count for each functionality matching the table.

- [ ] **Step 6: Write the labeling policy.** Create `docs/benchmark.md` with exactly this content. Task 14 appends the run/read sections.

````markdown
# Benchmark de moderación

## Para qué sirve

Antes de elegir un modelo de moderación los evaluamos a todos con el mismo examen: mensajes en español, portugués e inglés con la respuesta correcta ya etiquetada. El resultado (`bench_out/scoreboard.md`) muestra qué tan bien separa cada modelo los mensajes que hay que moderar de los que no, por fuente y por categoría.

La parte más importante del examen es la **suite propia** (`suites/es_casino.csv`): mensajes como los que vamos a ver en el chat del casino, con lunfardo, jerga de póker y trampas. Una fila cuenta recién cuando una persona revisó sus etiquetas.

## Categorías

| Columna | Es 1 cuando… | Ejemplos |
|---|---|---|
| `insult` | insulta, ridiculiza o acosa a una persona | "sos un manco", "callate gil", "p3lotud0" |
| `threat` | amenaza con daño o violencia, o con escrachar | "te voy a romper la cara", "sé dónde vivís" |
| `identity_hate` | ataca a gente por un rasgo protegido (lista abajo) | "bolita de mierda", "andá a lavar los platos" |
| `sexual_harassment` | comentarios, pedidos o avances sexuales no deseados hacia una persona | "mandame fotos", "pasame tu número, linda" |
| `profanity` | tiene puteadas o lenguaje vulgar, dirigido o no | "qué mano de mierda", "la puta madre" |

**Rasgos protegidos** (`identity_hate`): origen étnico o color de piel, nacionalidad, religión, género, orientación sexual, identidad de género y discapacidad. Los insultos por clase social ("villero", "planero"), por el cuerpo ("gorda", "enano") o por la edad ("viejo choto") son `insult`, no `identity_hate`.

**`target`**: a quién ataca el mensaje, si es dañino.
- `player`: a una persona concreta ("sos un manco").
- `group`: a un grupo ("los bolivianos son…").
- `none`: a nadie, o el mensaje no es dañino.

**Valores:** `1` sí, `0` no, `?` no estamos seguros o no nos ponemos de acuerdo (no se evalúa), vacío = sin revisar.

### Casos difíciles

- **Trato amistoso:** "boludo, vení", "negro, arrancamos", "gordo, buena mano" van con todo en `0`. El mismo término usado con odio ("negro de mierda") es `identity_hate=1`.
- **Contar un insulto no es insultar:** "me dijo trolo, ya lo reporté" va en `0`.
- **Negaciones:** "no sos un idiota" va en `0`.
- **Puteadas sin destinatario:** "qué día de mierda" lleva `profanity=1` y el resto en `0`.
- **Autocrítica:** "soy un manco" va en `0`.
- **Plata y apuestas:** "all in 500", "pago x2", "gg wp" van con todo en `0`. Ojo con los números que parecen leetspeak.
- **Lo ofuscado cuenta igual:** "p e l o t u d o" o "tr0l0" se etiquetan como si estuvieran bien escritos.
- **Mezcla de idiomas:** "nice hand manco" es `insult=1`.
- **Sarcasmo:** "qué genio que sos" va en `0`, salvo que ridiculice a alguien de forma clara.

## Cómo revisar la suite

1. Abrí `suites/es_casino.csv` en Excel o Google Sheets. Está separado por `;`.
2. En cada fila, leé el texto, corregí las categorías y `target`, y poné `reviewed` en `1`. Las sugerencias vienen de un generador: no las des por buenas.
3. Si una fila no tiene sentido, borrala.
4. **Sumá mensajes reales de memoria:** al menos 50 frases que hayas visto en chats de juegos o de póker, con `author` en `human`. Son las más valiosas, y el reporte las separa de las generadas.
5. Guardá como CSV (UTF-8). Sirve tanto con `;` como con `,`. Ojo: Excel convierte en fórmula un texto que empieza con `=`, `+` o `-`; si te pasa, poné un espacio adelante.
6. **Segunda opinión:** que otra persona revise una copia de las mismas ~150 filas sin mirar la primera revisión, y compará las dos:
   `uv run python scripts/bench.py agreement suites/es_casino.csv copia.csv`
   Si el kappa de una categoría da menos de 0.6, el criterio no está claro: hablenlo y ajusten esta guía.
````

- [ ] **Step 7: Run the tests again.**

Run: `uv run pytest tests/test_bench_seed.py tests/test_bench_suite.py -v`
Expected: all pass.

- [ ] **Step 8: Commit.** Summary: "add Spanish casino suite draft and labeling policy".

```bash
git add bench/seed.py scripts/seed_suite.py suites/ docs/benchmark.md tests/test_bench_seed.py
git commit
```

---

### Task 6: HateCheck source

**Files:**
- Create: `bench/sources/hatecheck.py`
- Test: `tests/test_bench_hatecheck.py`; create `tests/test_bench_slow.py` with the first slow test

**Interfaces:**
- Consumes: `bench.taxonomy.Example`.
- Produces:
  - `HATECHECK_REPOS`
  - `ALL_FUNCTIONALITIES` (29 names)
  - `hatecheck_example(row: dict, lang: str) -> Example | None`
  - `load_hatecheck(lang: str) -> list[Example]`
- Source names are `hatecheck-es`, `hatecheck-pt` and `hatecheck-en`.

Verified schema:
- **es/pt:** `mhc_case_id` str, `functionality`, `test_case`, `label_gold` (`hateful`/`non-hateful`), `templ_id` int, `disagreement_in_case` bool, in a single `test` split.
- **en:** `case_id` int instead of `mhc_case_id`, no `disagreement_*` columns, plus `slur_homonym_nh` and `slur_reclaimed_nh`.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_hatecheck.py`:

```python
import pytest

from bench.sources.hatecheck import ALL_FUNCTIONALITIES, hatecheck_example
from bench.taxonomy import FLAG, TARGETS_PLAYER, gold


def row(functionality, label=None, **extra):
    label = label or ("hateful" if functionality.endswith("_h") else "non-hateful")
    return {"mhc_case_id": "spanish-1", "functionality": functionality, "test_case": " Odio a los negros. ",
            "label_gold": label, "templ_id": 7, "disagreement_in_case": False, **extra}


def test_hateful_case_is_identity_hate_and_leaves_other_categories_unknown():
    e = hatecheck_example(row("derog_neg_emote_h"), "es")

    assert (e.id, e.source, e.lang, e.text) == ("hatecheck-es:spanish-1", "hatecheck-es", "es", "Odio a los negros.")
    assert e.labels == {"identity_hate": True}
    assert gold(e, FLAG) is True and gold(e, "insult") is None
    assert (e.functionality, e.cluster) == ("derog_neg_emote_h", "hatecheck-es:t7")


def test_threat_and_profanity_functionalities_set_those_categories_too():
    assert hatecheck_example(row("threat_dir_h"), "es").labels["threat"] is True
    assert hatecheck_example(row("profanity_h"), "es").labels["profanity"] is True


def test_untargeted_profanity_is_a_flag_negative_with_profanity():
    e = hatecheck_example(row("profanity_nh"), "es")

    assert gold(e, FLAG) is False and gold(e, "profanity") is True and e.target == "none"


def test_abuse_at_an_individual_is_an_insult_at_a_player():
    e = hatecheck_example(row("target_indiv_nh"), "es")

    assert gold(e, FLAG) is True and gold(e, TARGETS_PLAYER) is True and gold(e, "identity_hate") is False


@pytest.mark.parametrize("functionality", ["counter_quote_nh", "counter_ref_nh", "target_group_nh", "slur_reclaimed_nh"])
def test_non_hateful_cases_that_only_rule_out_hate_leave_flag_unknown(functionality):
    e = hatecheck_example(row(functionality), "es")

    assert gold(e, "identity_hate") is False and gold(e, FLAG) is None


def test_negated_hate_is_clean():
    assert gold(hatecheck_example(row("negate_neg_nh"), "es"), FLAG) is False


@pytest.mark.parametrize("value", [True, "True", "true"])
def test_cases_annotators_disagreed_on_are_dropped(value):
    assert hatecheck_example(row("slur_h", disagreement_in_case=value), "es") is None


def test_english_rows_use_case_id():
    english = {"case_id": 12, "functionality": "slur_h", "test_case": "I hate women. ", "label_gold": "hateful", "templ_id": 3}

    e = hatecheck_example(english, "en")

    assert (e.id, e.text) == ("hatecheck-en:12", "I hate women.")


def test_unknown_functionality_or_label_raises_naming_it():
    with pytest.raises(ValueError, match="new_thing_nh"):
        hatecheck_example(row("new_thing_nh"), "es")
    with pytest.raises(ValueError, match="hateful"):
        hatecheck_example(row("slur_h", label="non-hateful"), "es")
    with pytest.raises(ValueError, match="maybe"):
        hatecheck_example(row("slur_h", label="maybe"), "es")


@pytest.mark.parametrize("functionality", ALL_FUNCTIONALITIES)
def test_every_known_functionality_maps(functionality):
    assert hatecheck_example(row(functionality), "en") is not None
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_hatecheck.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** Create `bench/sources/hatecheck.py`:

```python
"""HateCheck (en) and Multilingual HateCheck (es, pt): functional tests of hate speech detection.

Each functionality only sets the categories it determines; the rest stay unknown. HateCheck's
"abuse at an individual" is an insult at a player under our policy, while abuse at non-protected
groups or objects only rules out identity hate.
"""

from __future__ import annotations

from bench.taxonomy import Example

HATECHECK_REPOS = {"es": "Paul/hatecheck-spanish", "pt": "Paul/hatecheck-portuguese", "en": "Paul/hatecheck"}
HATEFUL = (
    "derog_neg_emote_h", "derog_neg_attrib_h", "derog_dehum_h", "derog_impl_h", "threat_dir_h", "threat_norm_h",
    "slur_h", "profanity_h", "ref_subs_clause_h", "ref_subs_sent_h", "negate_pos_h", "phrase_question_h",
    "phrase_opinion_h", "spell_char_swap_h", "spell_char_del_h", "spell_space_del_h", "spell_space_add_h", "spell_leet_h",
)
NOT_HATEFUL = (
    "profanity_nh", "negate_neg_nh", "ident_neutral_nh", "ident_pos_nh", "counter_quote_nh", "counter_ref_nh",
    "target_obj_nh", "target_indiv_nh", "target_group_nh", "slur_homonym_nh", "slur_reclaimed_nh",
)
ALL_FUNCTIONALITIES = HATEFUL + NOT_HATEFUL
NO_ATTACK = {"insult": False, "threat": False, "identity_hate": False}


def _labels(functionality: str, hateful: bool) -> tuple[dict, str | None]:
    if hateful:
        labels = {"identity_hate": True}
        if functionality in ("threat_dir_h", "threat_norm_h"):
            labels["threat"] = True
        if functionality == "profanity_h":
            labels["profanity"] = True
        return labels, None
    if functionality == "profanity_nh":
        return {**NO_ATTACK, "profanity": True}, "none"
    if functionality in ("negate_neg_nh", "ident_neutral_nh", "ident_pos_nh"):
        return dict(NO_ATTACK), "none"
    if functionality == "target_obj_nh":
        return {"insult": False, "identity_hate": False}, None
    if functionality == "target_indiv_nh":
        return {"insult": True, "identity_hate": False}, "player"
    # Counter-speech can call someone a bigot, and abuse at non-protected groups is not ruled clean
    # by HateCheck, so only identity hate is known.
    return {"identity_hate": False}, None


def hatecheck_example(row: dict, lang: str) -> Example | None:
    if row.get("disagreement_in_case") in (True, "True", "true"):
        return None
    functionality, label = row["functionality"], row["label_gold"]
    if functionality not in ALL_FUNCTIONALITIES:
        raise ValueError(f"HateCheck functionality {functionality!r} has no mapping")
    if label not in ("hateful", "non-hateful"):
        raise ValueError(f"unexpected HateCheck label {label!r}")
    hateful = label == "hateful"
    if hateful != (functionality in HATEFUL):
        raise ValueError(f"{functionality!r} labeled {label!r}: the suffix says otherwise")
    labels, target = _labels(functionality, hateful)
    source = f"hatecheck-{lang}"
    case_id = row.get("mhc_case_id") or row["case_id"]
    return Example(
        id=f"{source}:{case_id}",
        source=source,
        lang=lang,
        text=row["test_case"].strip(),
        labels=labels,
        target=target,
        functionality=functionality,
        cluster=f"{source}:t{row['templ_id']}",
    )


def load_hatecheck(lang: str) -> list[Example]:
    from datasets import load_dataset

    rows = load_dataset(HATECHECK_REPOS[lang], split="test")
    return [e for e in (hatecheck_example(r, lang) for r in rows) if e is not None]
```

- [ ] **Step 4: Run the tests again.**

Run: `uv run pytest tests/test_bench_hatecheck.py -v`
Expected: all pass.

- [ ] **Step 5: Add the slow test against the real data.** Create `tests/test_bench_slow.py`:

```python
"""Benchmark checks against real datasets and models. They download data, so they only run with
`uv run pytest -m slow`."""

from collections import Counter

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("lang,minimum", [("es", 3500), ("pt", 3400), ("en", 3700)])
def test_real_hatecheck_loads_and_maps_every_case(lang, minimum):
    from bench.sources.hatecheck import load_hatecheck

    examples = load_hatecheck(lang)

    assert len(examples) >= minimum
    assert len(Counter(e.functionality for e in examples)) >= 27
```

Run: `uv run pytest -m slow tests/test_bench_slow.py -v`
Expected: 3 passed. If a count falls short, print `Counter(e.functionality …)` and report the gap; don't lower the minimum.

- [ ] **Step 6: Commit.** Summary: "add HateCheck es/pt/en source".

```bash
git add bench/sources/hatecheck.py tests/test_bench_hatecheck.py tests/test_bench_slow.py
git commit
```

---

### Task 7: Portuguese sources (ToLD-Br, OLID-BR)

**Files:**
- Create: `bench/sources/portuguese.py`
- Test: `tests/test_bench_portuguese.py`; append to `tests/test_bench_slow.py`

**Interfaces:**
- Consumes: `bench.taxonomy.Example`.
- Produces:
  - `toldbr_example(row, index) -> Example`
  - `olidbr_example(row) -> Example`
  - `load_toldbr() -> list[Example]`
  - `load_olidbr() -> list[Example]`
- Sources are `toldbr` and `olidbr`, both `lang="pt"`.

Verified schemas:
- **`mteb/told-br`** (split `train`, 21,000 rows): `text` plus `homophobia`, `obscene`, `insult`, `racism`, `misogyny`, `xenophobia` = annotator votes 0–3. These may arrive as ints or as strings like `"2.0"`.
- **`dougtrajano/olid-br`** (split `test`, 1,738 rows):
  - `id` (hex)
  - `text`
  - `is_offensive` (`OFF`/`NOT`)
  - `is_targeted` (`TIN`/`UNT`)
  - `targeted_type` (`IND`/`GRP`/`OTH`/None)
  - booleans: `health`, `ideology`, `insult`, `lgbtqphobia`, `other_lifestyle`, `physical_aspects`, `profanity_obscene`, `racism`, `religious_intolerance`, `sexism`, `xenophobia`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_portuguese.py`:

```python
import pytest

from bench.sources.portuguese import olidbr_example, toldbr_example
from bench.taxonomy import FLAG, TARGETS_PLAYER, gold

TOLD_ZERO = {"text": " texto ", "homophobia": 0, "obscene": 0, "insult": 0, "racism": 0, "misogyny": 0, "xenophobia": 0}
OLID_FALSE = {c: False for c in ("health", "ideology", "insult", "lgbtqphobia", "other_lifestyle", "physical_aspects",
                                   "profanity_obscene", "racism", "religious_intolerance", "sexism", "xenophobia")}


def told(**votes):
    return toldbr_example({**TOLD_ZERO, **votes}, 5)


def olid(**fields):
    base = {"id": "ab12", "text": "texto", "is_offensive": "OFF", "is_targeted": "TIN", "targeted_type": "IND", **OLID_FALSE}
    return olidbr_example({**base, **fields})


def test_told_without_votes_is_clean():
    e = told()

    assert (e.id, e.lang, e.text) == ("toldbr:5", "pt", "texto")
    assert gold(e, FLAG) is False and gold(e, "profanity") is False and e.target == "none"


def test_told_obscene_only_is_profanity_but_not_flagged():
    e = told(obscene="3.0")

    assert gold(e, FLAG) is False and gold(e, "profanity") is True


def test_told_majority_insult_is_flagged_and_a_single_vote_is_unknown():
    assert gold(told(insult=2), FLAG) is True
    single = told(insult=1)
    assert gold(single, "insult") is None and gold(single, FLAG) is None


def test_told_identity_columns_combine():
    assert gold(told(racism=2), "identity_hate") is True
    assert gold(told(racism=1), "identity_hate") is None
    assert gold(told(insult=2), "identity_hate") is False


def test_told_vote_out_of_range_raises():
    with pytest.raises(ValueError, match="7"):
        told(insult=7)


def test_olid_not_offensive_is_clean():
    e = olid(is_offensive="NOT", is_targeted="UNT", targeted_type=None)

    assert gold(e, FLAG) is False and gold(e, "profanity") is False and e.id == "olidbr:ab12"


def test_olid_untargeted_swearing_only_is_a_flag_negative():
    e = olid(is_targeted="UNT", targeted_type=None, profanity_obscene=True)

    assert gold(e, FLAG) is False and gold(e, "profanity") is True


def test_olid_insult_at_an_individual_targets_a_player():
    e = olid(insult=True)

    assert gold(e, FLAG) is True and gold(e, TARGETS_PLAYER) is True


def test_olid_group_insult_is_not_a_personal_insult():
    e = olid(targeted_type="GRP", insult=True, racism=True)

    assert gold(e, "insult") is None and gold(e, "identity_hate") is True and e.target == "group"


def test_olid_health_is_ambiguous_for_identity_hate():
    assert gold(olid(health=True), "identity_hate") is None


@pytest.mark.parametrize("field,value", [("is_offensive", "MAYBE"), ("is_targeted", "X"), ("targeted_type", "ORG")])
def test_olid_unexpected_values_raise(field, value):
    with pytest.raises(ValueError, match=value):
        olid(**{field: value})
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_portuguese.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** Create `bench/sources/portuguese.py`:

```python
"""Brazilian Portuguese sources: ToLD-Br (annotator votes per category) and OLID-BR (offensive, target,
categories).

When annotators had the chance to call a message an attack and only marked swearing, or nothing,
it counts as clean. That gives `flag` its hard negatives: swearing that hurts no one.
"""

from __future__ import annotations

from bench.taxonomy import Example

TOLDBR_REPO = "mteb/told-br"
OLIDBR_REPO = "dougtrajano/olid-br"
TOLD_COLUMNS = ("homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia")
TOLD_IDENTITY = ("homophobia", "racism", "misogyny", "xenophobia")
OLID_IDENTITY = ("racism", "sexism", "lgbtqphobia", "xenophobia", "religious_intolerance")
OLID_OTHER = ("health", "ideology", "insult", "other_lifestyle", "physical_aspects")


def _votes(value) -> int:
    votes = int(float(value))
    if not 0 <= votes <= 3:
        raise ValueError(f"ToLD-Br vote count out of range: {value!r}")
    return votes


def _majority(votes: int) -> bool | None:
    """Three annotators: two or more is a yes, none is a no, a single vote is too contested to score."""
    return True if votes >= 2 else False if votes == 0 else None


def toldbr_example(row: dict, index: int) -> Example:
    votes = {c: _votes(row[c]) for c in TOLD_COLUMNS}
    identity = [votes[c] for c in TOLD_IDENTITY]
    if not any(votes[c] for c in TOLD_COLUMNS if c != "obscene"):
        labels = {"insult": False, "threat": False, "identity_hate": False, "profanity": _majority(votes["obscene"])}
        target = "none"
    else:
        labels = {
            "insult": _majority(votes["insult"]),
            "identity_hate": True if any(v >= 2 for v in identity) else False if not any(identity) else None,
            "profanity": _majority(votes["obscene"]),
        }
        target = None
    return Example(id=f"toldbr:{index}", source="toldbr", lang="pt", text=row["text"].strip(), labels=labels, target=target)


def olidbr_example(row: dict) -> Example:
    offensive, targeted, kind = row["is_offensive"], row["is_targeted"], row["targeted_type"]
    if offensive not in ("OFF", "NOT"):
        raise ValueError(f"unexpected OLID-BR is_offensive {offensive!r}")
    if targeted not in ("TIN", "UNT"):
        raise ValueError(f"unexpected OLID-BR is_targeted {targeted!r}")
    if kind not in ("IND", "GRP", "OTH", None):
        raise ValueError(f"unexpected OLID-BR targeted_type {kind!r}")

    if offensive == "NOT":
        labels = {"insult": False, "threat": False, "identity_hate": False, "profanity": False}
        target = "none"
    else:
        attack_categories = [c for c in OLID_IDENTITY + OLID_OTHER if row[c]]
        if targeted == "UNT" and not attack_categories:
            labels = {"insult": False, "threat": False, "identity_hate": False,
                      "profanity": True if row["profanity_obscene"] else None}
        else:
            identity = any(row[c] for c in OLID_IDENTITY)
            labels = {
                # OLID's insult includes groups; ours is aimed at a person. IND is often a public figure.
                "insult": None if kind == "GRP" else bool(row["insult"]),
                # `health` mixes disability (protected) with other health insults.
                "identity_hate": True if identity else (None if row["health"] else False),
                "profanity": bool(row["profanity_obscene"]),
            }
        target = "none" if targeted == "UNT" else {"IND": "player", "GRP": "group"}.get(kind)
    return Example(id=f"olidbr:{row['id']}", source="olidbr", lang="pt", text=row["text"].strip(), labels=labels, target=target)


def load_toldbr() -> list[Example]:
    from datasets import load_dataset

    return [toldbr_example(row, i) for i, row in enumerate(load_dataset(TOLDBR_REPO, split="train"))]


def load_olidbr() -> list[Example]:
    from datasets import load_dataset

    return [olidbr_example(row) for row in load_dataset(OLIDBR_REPO, split="test")]
```

- [ ] **Step 4: Run the tests again.**

Run: `uv run pytest tests/test_bench_portuguese.py -v`
Expected: all pass.

- [ ] **Step 5: Add the slow tests.** Append to `tests/test_bench_slow.py`:

```python
def test_real_toldbr_and_olidbr_load_with_both_flag_classes():
    from bench.sources.portuguese import load_olidbr, load_toldbr
    from bench.taxonomy import FLAG, gold

    told, olid = load_toldbr(), load_olidbr()

    assert len(told) == 21000 and len(olid) == 1738
    for examples in (told, olid):
        flags = Counter(gold(e, FLAG) for e in examples)
        assert flags[True] > 100 and flags[False] > 100, flags
```

Run: `uv run pytest -m slow tests/test_bench_slow.py -v`
Expected: all pass. If `mteb/told-br` fails to load, report BLOCKED with the error. The fallback is the official parquet conversion at `hf://datasets/JAugusto97/told-br@refs%2Fconvert%2Fparquet/multilabel/train/0000.parquet` via `load_dataset("parquet", data_files=...)`.

- [ ] **Step 6: Commit.** Summary: "add ToLD-Br and OLID-BR sources".

```bash
git add bench/sources/portuguese.py tests/test_bench_portuguese.py tests/test_bench_slow.py
git commit
```

### Task 8: Source registry, sampling and splits

**Files:**
- Modify: `bench/sources/__init__.py` (it was empty)
- Test: `tests/test_bench_sources.py`

**Interfaces:**
- Consumes:
  - `load_suite` (Task 4), `load_hatecheck` (Task 6), `load_toldbr` and `load_olidbr` (Task 7)
  - `bench.taxonomy.has_gold`
- Produces:
  - `Source(name, lang, license, load: Callable[[bool], list[Example]], sampled=False)`
  - `SOURCES: dict[str, Source]`
  - `DEFAULT_MAX_PER_SOURCE = 4000`
  - `stable_hash(text) -> int`
  - `sample(examples, max_n) -> list[Example]`
  - `split_of(example) -> "calib" | "eval"`
  - `load_sources(names, max_per_source=4000, include_unreviewed=False, sources=None, log=print) -> list[Example]`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_sources.py`:

```python
import pytest

from bench.sources import SOURCES, Source, load_sources, sample, split_of
from bench.taxonomy import Example


def ex(i, labels=None, cluster=None):
    return Example(id=f"s:{i}", source="s", lang="es", text=f"texto {i}",
                   labels={"insult": True} if labels is None else labels, cluster=cluster)


def test_sample_drops_examples_with_nothing_to_score_and_is_stable():
    examples = [ex(i) for i in range(50)] + [ex(f"x{i}", labels={}) for i in range(10)]

    picked = sample(examples, 20)

    assert len(picked) == 20 and all(e.labels for e in picked)
    assert [e.id for e in sample(list(reversed(examples)), 20)] == [e.id for e in picked]
    assert len(sample(examples, 1000)) == 50


def test_a_fifth_goes_to_calibration_and_clusters_stay_together():
    splits = [split_of(ex(i)) for i in range(5000)]

    assert 0.17 < splits.count("calib") / 5000 < 0.23
    assert len({split_of(ex(i, cluster="tmpl-1")) for i in range(30)}) == 1


def test_only_sampled_sources_are_capped_and_an_empty_source_warns():
    logs = []
    sources = {
        "big": Source("big", "es", "x", lambda _: [ex(i) for i in range(30)], sampled=True),
        "whole": Source("whole", "es", "x", lambda _: [ex(f"w{i}") for i in range(30)]),
        "empty": Source("empty", "es", "x", lambda _: [ex("e", labels={})]),
    }

    examples = load_sources(["big", "whole", "empty"], max_per_source=10, sources=sources, log=logs.append)

    assert len(examples) == 40
    assert any("empty" in line for line in logs)


def test_include_unreviewed_reaches_the_loader():
    seen = []
    sources = {"suite": Source("suite", "es", "x", lambda unreviewed: seen.append(unreviewed) or [ex(1)])}

    load_sources(["suite"], include_unreviewed=True, sources=sources, log=lambda _: None)

    assert seen == [True]


def test_unknown_source_is_an_error():
    with pytest.raises(ValueError, match="desconocida"):
        load_sources(["nope"], sources={}, log=lambda _: None)


def test_registry_has_the_planned_sources():
    assert set(SOURCES) == {"suite-es", "hatecheck-es", "hatecheck-pt", "hatecheck-en", "toldbr", "olidbr"}
    assert [n for n, s in SOURCES.items() if s.sampled] == ["toldbr"]
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_sources.py -v`
Expected: FAIL with `ImportError: cannot import name 'SOURCES'`.

- [ ] **Step 3: Implement.** Replace `bench/sources/__init__.py` with:

```python
"""Every benchmark source by name, and how examples are sampled and split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bench.sources.hatecheck import load_hatecheck
from bench.sources.portuguese import load_olidbr, load_toldbr
from bench.sources.suite import load_suite
from bench.taxonomy import Example, has_gold

SUITES_DIR = Path(__file__).resolve().parents[2] / "suites"
DEFAULT_MAX_PER_SOURCE = 4000
CALIB_ONE_IN = 5


@dataclass(frozen=True)
class Source:
    name: str
    lang: str
    license: str
    # Called with include_unreviewed; only the suite uses it.
    load: Callable[[bool], list[Example]]
    # Large corpora are sampled; functional tests and the suite are used whole.
    sampled: bool = False


SOURCES = {s.name: s for s in (
    Source("suite-es", "es", "propia",
           lambda unreviewed: load_suite(SUITES_DIR / "es_casino.csv", "suite-es", unreviewed)),
    Source("hatecheck-es", "es", "CC BY 4.0", lambda _: load_hatecheck("es")),
    Source("hatecheck-pt", "pt", "CC BY 4.0", lambda _: load_hatecheck("pt")),
    Source("hatecheck-en", "en", "CC BY 4.0", lambda _: load_hatecheck("en")),
    Source("toldbr", "pt", "CC BY-SA 4.0", lambda _: load_toldbr(), sampled=True),
    Source("olidbr", "pt", "CC BY 4.0", lambda _: load_olidbr()),
)}


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)


def sample(examples: list[Example], max_n: int) -> list[Example]:
    """Up to `max_n` examples that have something to score, the same ones on every run."""
    keep = [e for e in examples if has_gold(e)]
    if len(keep) <= max_n:
        return keep
    return sorted(keep, key=lambda e: stable_hash(e.id))[:max_n]


def split_of(example: Example) -> str:
    """`calib` (operating thresholds are chosen here) or `eval` (what the report shows).

    Decided by cluster, so a HateCheck template never lands on both sides."""
    return "calib" if stable_hash(example.cluster_key) % CALIB_ONE_IN == 0 else "eval"


def load_sources(names, max_per_source: int = DEFAULT_MAX_PER_SOURCE, include_unreviewed: bool = False,
                 sources: dict | None = None, log=print) -> list[Example]:
    sources = SOURCES if sources is None else sources
    unknown = [n for n in names if n not in sources]
    if unknown:
        raise ValueError(f"fuente desconocida: {', '.join(unknown)}. Opciones: {', '.join(sources)}")
    examples = []
    for name in names:
        source = sources[name]
        loaded = source.load(include_unreviewed)
        loaded = sample(loaded, max_per_source) if source.sampled else [e for e in loaded if has_gold(e)]
        if not loaded:
            log(f"Aviso: {name} no tiene ejemplos con etiquetas (¿la suite todavía no está revisada?). Se saltea.")
        examples.extend(loaded)
    return examples
```

- [ ] **Step 4: Run the tests again.**

Run: `uv run pytest tests/test_bench_sources.py tests/test_bench_seed.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.** Summary: "add source registry, sampling and calibration split".

```bash
git add bench/sources/__init__.py tests/test_bench_sources.py
git commit
```

---

### Task 9: Contestant protocol, prediction cache and runner

**Files:**
- Create: `bench/contestants/__init__.py` (empty for now; Task 11 fills it), `bench/contestants/base.py`, `bench/cache.py`, `bench/runner.py`
- Test: `tests/test_bench_runner.py`

**Interfaces:**
- Consumes: `bench.taxonomy.FLAG`, `FLAG_CATEGORIES`, `Example`.
- Produces:
  - `Contestant` protocol, with attributes `name`, `version`, `languages: frozenset[str]`, `notes` and `license` (an SPDX id), and methods `predict(texts, lang) -> list[dict[str, float]]` and `to_scores(raw) -> dict[str, float]`
  - `effective_scores(contestant, raw) -> dict[str, float]`, which adds `flag` = the max of the three core categories when all are present and `flag` is missing
  - `map_labels(raw, mapping) -> dict[str, float]`
  - `free_gpu() -> None`
  - an optional `unload()` method on contestants
  - `text_hash(text) -> str`
  - `PredictionCache(root)` with `.load(contestant, source) -> dict[id, (text_sha, raw)]` and `.append(contestant, source, rows: list[(id, text_sha, raw)])`
  - `run(contestants, examples, cache, batch_size=32, log=print) -> dict[contestant_name, dict[example_id, raw]]`, which calls `contestant.unload()`, when defined, once each contestant is done

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_runner.py`:

```python
import pytest

from bench.cache import PredictionCache, text_hash
from bench.contestants.base import effective_scores, map_labels
from bench.runner import run
from bench.taxonomy import Example

QUIET = {"log": lambda _: None}


class FakeContestant:
    name = "fake"
    version = "1"
    notes = ""

    def __init__(self, languages=("es", "pt"), fail_on_call=None):
        self.languages = frozenset(languages)
        self.calls = []
        self.fail_on_call = fail_on_call

    def predict(self, texts, lang):
        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            raise ConnectionError("se cortó la conexión")
        return [{"len": float(len(t))} for t in texts]

    def to_scores(self, raw):
        return {"insult": raw["len"]}


def ex(i, text=None, lang="es", source="s"):
    return Example(id=f"{source}:{i}", source=source, lang=lang, text=text or f"mensaje {i}", labels={"insult": True})


def test_predicts_every_example_once_in_batches(tmp_path):
    contestant = FakeContestant()

    raw = run([contestant], [ex(i) for i in range(5)], PredictionCache(tmp_path), batch_size=2, **QUIET)

    assert [len(batch) for batch in contestant.calls] == [2, 2, 1]
    assert raw["fake"]["s:3"] == {"len": float(len("mensaje 3"))}


def test_a_second_run_comes_from_the_cache(tmp_path):
    examples = [ex(i) for i in range(3)]
    run([FakeContestant()], examples, PredictionCache(tmp_path), **QUIET)
    again = FakeContestant()

    raw = run([again], examples, PredictionCache(tmp_path), **QUIET)

    assert again.calls == [] and len(raw["fake"]) == 3


def test_an_edited_text_is_predicted_again(tmp_path):
    run([FakeContestant()], [ex(1), ex(2)], PredictionCache(tmp_path), **QUIET)
    again = FakeContestant()

    raw = run([again], [ex(1), ex(2, text="texto corregido")], PredictionCache(tmp_path), **QUIET)

    assert again.calls == [["texto corregido"]]
    assert raw["fake"]["s:2"] == {"len": float(len("texto corregido"))}


def test_a_crashed_run_resumes_after_the_last_finished_batch(tmp_path):
    examples = [ex(i) for i in range(4)]
    with pytest.raises(ConnectionError):
        run([FakeContestant(fail_on_call=2)], examples, PredictionCache(tmp_path), batch_size=2, **QUIET)
    again = FakeContestant()

    run([again], examples, PredictionCache(tmp_path), batch_size=2, **QUIET)

    assert again.calls == [["mensaje 2", "mensaje 3"]]


def test_a_truncated_last_line_is_ignored_and_does_not_corrupt_the_next_append(tmp_path):
    cache, contestant = PredictionCache(tmp_path), FakeContestant()
    run([contestant], [ex(1)], cache, **QUIET)
    path = next(tmp_path.rglob("*.jsonl"))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"id": "s:2", "text_sh')

    assert set(cache.load(contestant, "s")) == {"s:1"}
    run([FakeContestant()], [ex(1), ex(2)], cache, **QUIET)
    assert set(cache.load(contestant, "s")) == {"s:1", "s:2"}


def test_languages_the_contestant_does_not_support_are_skipped(tmp_path):
    raw = run([FakeContestant(languages=("es",))], [ex(1), ex(2, lang="en", source="e")], PredictionCache(tmp_path), **QUIET)

    assert set(raw["fake"]) == {"s:1"}


def test_the_wrong_number_of_outputs_is_an_error(tmp_path):
    contestant = FakeContestant()
    contestant.predict = lambda texts, lang: []

    with pytest.raises(RuntimeError, match="fake"):
        run([contestant], [ex(1)], PredictionCache(tmp_path), **QUIET)


def test_cache_folder_name_is_safe_on_windows(tmp_path):
    contestant = FakeContestant()
    contestant.version = "rev:abc/def"

    PredictionCache(tmp_path).append(contestant, "s", [("s:1", text_hash("x"), {"len": 1.0})])

    assert [p.parent.name for p in tmp_path.rglob("*.jsonl")] == ["fake@rev_abc_def"]


def test_each_contestant_is_unloaded_once_it_is_done(tmp_path):
    events = []

    class Unloadable(FakeContestant):
        def predict(self, texts, lang):
            events.append("predict")
            return super().predict(texts, lang)

        def unload(self):
            events.append("unload")

    run([Unloadable()], [ex(1), ex(2, lang="pt", source="p")], PredictionCache(tmp_path), **QUIET)

    assert events == ["predict", "predict", "unload"]


class Fixed:
    def __init__(self, scores):
        self.scores = scores

    def to_scores(self, raw):
        return self.scores


def test_flag_is_the_max_of_insult_threat_and_identity_hate_only_when_all_three_are_present():
    assert effective_scores(Fixed({"insult": 0.2, "threat": 0.9, "identity_hate": 0.1}), {})["flag"] == 0.9
    assert "flag" not in effective_scores(Fixed({"identity_hate": 0.7}), {})
    assert effective_scores(Fixed({"flag": 0.3, "insult": 0.9, "threat": 0.9, "identity_hate": 0.9}), {})["flag"] == 0.3


def test_map_labels_renames_only_labels_that_are_present():
    assert map_labels({"insult": 0.5, "toxicity": 0.9}, {"insult": "insult", "threat": "threat"}) == {"insult": 0.5}
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.cache'`.

- [ ] **Step 3: Implement.** Create an empty `bench/contestants/__init__.py`, then `bench/contestants/base.py`:

```python
"""What every contestant provides, and how its scores become the taxonomy's."""

from __future__ import annotations

from typing import Protocol

from bench.taxonomy import FLAG, FLAG_CATEGORIES


class Contestant(Protocol):
    name: str
    # Part of the cache key: change it whenever the same text could get a different raw output.
    version: str
    languages: frozenset[str]
    # Shown in the report: what is approximate or risky about this contestant.
    notes: str
    # SPDX id of the model's license. Only open, self-hostable models: see bench.contestants.OPEN_LICENSES.
    license: str

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        """Raw outputs under the model's own label names. Loads the model on first use."""

    def to_scores(self, raw: dict[str, float]) -> dict[str, float]:
        """Raw outputs mapped onto taxonomy keys. Pure, so scoring from the cache needs no model."""

    # Optional `unload()`: drop the loaded model. The runner calls it once a contestant is done, so the
    # next contestant fits on a small GPU.


def effective_scores(contestant, raw: dict[str, float]) -> dict[str, float]:
    scores = dict(contestant.to_scores(raw))
    if FLAG not in scores and all(c in scores for c in FLAG_CATEGORIES):
        scores[FLAG] = max(scores[c] for c in FLAG_CATEGORIES)
    return scores


def map_labels(raw: dict[str, float], mapping: dict[str, str]) -> dict[str, float]:
    """{our key: raw[their label]} for the labels `mapping` names and `raw` has."""
    return {ours: float(raw[theirs]) for theirs, ours in mapping.items() if theirs in raw}


def free_gpu() -> None:
    """Give back the GPU memory of a model that was just dropped, so the next one fits."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

Create `bench/cache.py`:

```python
"""Raw contestant outputs on disk: one JSONL per contestant version and source.

Outputs are stored before any mapping onto the taxonomy, so fixing a mapping needs no new inference.
Each line also stores a hash of the text: when a reviewer edits a suite row, its old output is ignored.
Lines are appended after every batch, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class PredictionCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, contestant, source: str) -> Path:
        folder = re.sub(r"[^A-Za-z0-9._@-]", "_", f"{contestant.name}@{contestant.version}")
        return self.root / folder / f"{source}.jsonl"

    def load(self, contestant, source: str) -> dict[str, tuple[str, dict]]:
        path = self._path(contestant, source)
        if not path.exists():
            return {}
        cached = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a run killed mid-write leaves a partial line
            cached[row["id"]] = (row["text_sha"], row["raw"])
        return cached

    def append(self, contestant, source: str, rows: list[tuple[str, str, dict]]) -> None:
        path = self._path(contestant, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = False
        if path.exists() and path.stat().st_size:
            with path.open("rb") as f:
                f.seek(-1, 2)
                needs_newline = f.read(1) != b"\n"
        with path.open("a", encoding="utf-8", newline="\n") as f:
            if needs_newline:
                f.write("\n")
            for example_id, sha, raw in rows:
                f.write(json.dumps({"id": example_id, "text_sha": sha, "raw": raw}, ensure_ascii=False) + "\n")
```

Create `bench/runner.py`:

```python
"""Runs every contestant over every example, through the prediction cache."""

from __future__ import annotations

from collections import defaultdict

from bench.cache import PredictionCache, text_hash


def run(contestants, examples, cache: PredictionCache, batch_size: int = 32, log=print) -> dict[str, dict[str, dict]]:
    groups = defaultdict(list)
    for e in examples:
        groups[(e.source, e.lang)].append(e)

    raw: dict[str, dict[str, dict]] = {}
    for contestant in contestants:
        raw[contestant.name] = {}
        for (source, lang), group in sorted(groups.items()):
            if lang not in contestant.languages:
                log(f"{contestant.name}: no soporta '{lang}', se saltea {source}.")
                continue
            cached = cache.load(contestant, source)
            todo = [e for e in group if cached.get(e.id, ("",))[0] != text_hash(e.text)]
            log(f"{contestant.name} · {source}: {len(todo)} mensajes nuevos, {len(group) - len(todo)} en caché.")
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                outputs = contestant.predict([e.text for e in batch], lang)
                if len(outputs) != len(batch):
                    raise RuntimeError(f"{contestant.name} devolvió {len(outputs)} resultados para {len(batch)} textos")
                rows = [(e.id, text_hash(e.text), out) for e, out in zip(batch, outputs)]
                cache.append(contestant, source, rows)
                cached.update({example_id: (sha, out) for example_id, sha, out in rows})
            raw[contestant.name].update({e.id: cached[e.id][1] for e in group})
        unload = getattr(contestant, "unload", None)
        if unload is not None:
            unload()
    return raw
```

- [ ] **Step 4: Run the tests again.**

Run: `uv run pytest tests/test_bench_runner.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.** Summary: "add contestant protocol, prediction cache and runner".

```bash
git add bench/contestants/__init__.py bench/contestants/base.py bench/cache.py bench/runner.py tests/test_bench_runner.py
git commit
```

---

### Task 10: Laya contestants

**Files:**
- Modify: `app/classifier.py`, the `build_router` signature and its `Router(...)` call
- Create: `bench/contestants/laya.py`
- Test: `tests/test_bench_laya.py`; add one test to `tests/test_routing.py`; append to `tests/test_bench_slow.py`

**Interfaces:**
- Consumes:
  - `app.classifier.QUESTIONS`, `STATE_KEY`, `build_router(checkpoint=None, device=None)`
  - `bench.taxonomy`
- Produces:
  - `TAXONOMY_QUESTIONS`, `POSITIVE`
  - `fingerprint(questions) -> str`
  - `flatten(result) -> dict[str, float]` with keys `"<question>.<option>"` plus `routed_english`
  - `taxonomy_scores(raw)`, `app_scores(raw)`
  - `LayaContestant(name, questions, scores, notes, device=None, router=None)`
  - `laya_taxonomy(device=None, router=None)`, `laya_app(device=None, router=None)`
  - `LayaContestant.unload()`

Verified Laya 0.3.22 API:
- `Router.predict_batch(requests)` takes `requests = [{"state": {...}, "questions": {...}}]` and returns results in input order.
- Each result has `answers[qid]["probabilities"]` (option → p) and `routing["model"]` (`"english"` or `"multilingual"`).
- `Router.preload(names)` loads the named checkpoints.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_laya.py`:

```python
import pytest

import bench.contestants.laya as laya_module
from bench.contestants.base import effective_scores
from bench.contestants.laya import (
    POSITIVE, TAXONOMY_QUESTIONS, LayaContestant, flatten, laya_app, laya_taxonomy, taxonomy_scores,
)
from bench.taxonomy import CATEGORIES, FLAG, TARGETS_PLAYER


def answer(options, first=0.8):
    rest = (1 - first) / (len(options) - 1)
    return {"type": "choice", "choice": options[0], "confidence": 0.4, "answer_confidence": first,
            "probabilities": {o: first if i == 0 else rest for i, o in enumerate(options)}, "action": {"act_probability": 0.5}}


def result(questions, model="multilingual"):
    return {"answers": {qid: answer(list(q["criteria"])) for qid, q in questions.items()},
            "routing": {"model": model, "repo": "r", "reason": "x"}, "usage": {}}


class FakeRouter:
    def __init__(self, model="multilingual"):
        self.batches, self.preloaded, self.model = [], None, model

    def preload(self, names):
        self.preloaded = names
        return self

    def predict_batch(self, requests, **kwargs):
        self.batches.append(requests)
        return [result(r["questions"], self.model) for r in requests]


def test_predict_sends_one_request_per_message_in_one_batch():
    router = FakeRouter()

    raw = laya_taxonomy(router=router).predict(["sos un manco", "gg"], "es")

    assert router.batches == [[
        {"state": {"message": "sos un manco"}, "questions": TAXONOMY_QUESTIONS},
        {"state": {"message": "gg"}, "questions": TAXONOMY_QUESTIONS},
    ]]
    assert raw[0]["insult.insult"] == 0.8 and raw[0]["routed_english"] == 0.0


def test_english_routing_is_recorded():
    assert laya_taxonomy(router=FakeRouter(model="english")).predict(["you suck"], "en")[0]["routed_english"] == 1.0


def test_taxonomy_scores_read_the_positive_option_of_each_question():
    raw = flatten(result(TAXONOMY_QUESTIONS))

    scores = taxonomy_scores(raw)

    assert set(scores) == set(CATEGORIES) | {TARGETS_PLAYER}
    assert scores["identity_hate"] == raw["identity_hate.hate"] == 0.8
    assert scores[TARGETS_PLAYER] == raw["target.player"]


def test_flag_is_the_max_of_the_core_categories():
    raw = {**flatten(result(TAXONOMY_QUESTIONS)), "threat.threat": 0.95}

    assert effective_scores(laya_taxonomy(router=FakeRouter()), raw)[FLAG] == 0.95


def test_app_contestant_asks_the_apps_questions_and_uses_insult_as_flag():
    from app.classifier import QUESTIONS

    router = FakeRouter()
    contestant = laya_app(router=router)
    [raw] = contestant.predict(["sos un idiota"], "es")

    assert router.batches[0][0]["questions"] == QUESTIONS
    assert effective_scores(contestant, raw) == {FLAG: raw["insult.insult"]}


def test_version_follows_the_questions():
    same = LayaContestant("x", TAXONOMY_QUESTIONS, taxonomy_scores, "", router=FakeRouter())
    changed = {**TAXONOMY_QUESTIONS, "threat": {**TAXONOMY_QUESTIONS["threat"], "instructions": "Is `message` a threat?"}}

    assert same.version == LayaContestant("x", TAXONOMY_QUESTIONS, taxonomy_scores, "", router=FakeRouter()).version
    assert same.version != LayaContestant("x", changed, taxonomy_scores, "", router=FakeRouter()).version


def test_router_is_built_lazily_once_with_the_device_and_both_checkpoints(monkeypatch):
    built = []

    def fake_build_router(device=None):
        built.append(device)
        return FakeRouter()

    monkeypatch.setattr(laya_module, "build_router", fake_build_router)
    contestant = laya_taxonomy(device="cpu")
    assert built == []

    contestant.predict(["hola"], "es")
    contestant.predict(["chau"], "es")

    assert built == ["cpu"]
    assert contestant._router.preloaded == ["english", "multilingual"]


def test_unload_drops_the_router_and_the_next_prediction_rebuilds_it(monkeypatch):
    built = []
    monkeypatch.setattr(laya_module, "build_router", lambda device=None: built.append(device) or FakeRouter())
    contestant = laya_taxonomy()

    contestant.predict(["hola"], "es")
    contestant.unload()
    contestant.predict(["chau"], "es")

    assert len(built) == 2


@pytest.mark.parametrize("qid", list(TAXONOMY_QUESTIONS))
def test_every_question_refers_to_the_message_and_has_two_or_three_options(qid):
    question = TAXONOMY_QUESTIONS[qid]

    assert "`message`" in question["instructions"]
    assert 2 <= len(question["criteria"]) <= 3


def test_every_positive_option_exists():
    for category, option in POSITIVE.items():
        assert option in TAXONOMY_QUESTIONS[category]["criteria"]
```

Append to `tests/test_routing.py`:

```python
def test_explicit_device_overrides_the_environment(monkeypatch):
    import laya

    seen = {}
    monkeypatch.setattr(laya, "Router", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setenv("LAYA_DEVICE", "cuda")

    build_router(device="cpu")

    assert seen["device"] == "cpu"
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_laya.py tests/test_routing.py -v`
Expected: `test_bench_laya.py` FAILS with `ModuleNotFoundError`; `test_explicit_device_overrides_the_environment` FAILS with `TypeError: build_router() got an unexpected keyword argument 'device'`.

- [ ] **Step 3: Add the device parameter.** In `app/classifier.py`, change `build_router`. The docstring keeps its first lines and gains the `device` sentence:

```python
def build_router(checkpoint: Path | None = None, device: str | None = None):
    """A Router with no checkpoint loaded yet; routing decisions work without loading one.

    `checkpoint` is a local fine-tuned directory that answers in place of `multilingual`.
    `device` overrides LAYA_DEVICE (the benchmark picks it per run).
    """
```

and its last line:

```python
    return Router(models=models, device=device or os.environ.get("LAYA_DEVICE") or None, default="multilingual")
```

- [ ] **Step 4: Implement the contestants.** Create `bench/contestants/laya.py`:

```python
"""Laya answering the taxonomy as typed questions, zero-shot, with the same Router setup as the chat
app: text Laya identifies as English goes to the English checkpoint, the rest to the multilingual one."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata

from app.classifier import QUESTIONS as APP_QUESTIONS
from app.classifier import STATE_KEY, build_router
from bench.contestants.base import free_gpu
from bench.taxonomy import FLAG, LANGS, TARGETS_PLAYER

# One question per category, positive option first. A first wording, not tuned: the report says so.
TAXONOMY_QUESTIONS = {
    "insult": {
        "type": "choice",
        "instructions": "Does `message` insult, mock or harass a specific person?",
        "criteria": {
            "insult": "insults, name-calling, mockery or harassment aimed at a person",
            "no_insult": "no insult aimed at a person; includes friendly banter, self-criticism, swearing at no one and criticism of play",
        },
    },
    "threat": {
        "type": "choice",
        "instructions": "Does `message` threaten someone?",
        "criteria": {
            "threat": "threatens violence or harm against a person or group, or to expose where they live",
            "no_threat": "no threat of violence or harm",
        },
    },
    "identity_hate": {
        "type": "choice",
        "instructions": "Does `message` attack people for who they are?",
        "criteria": {
            "hate": "attacks, demeans or slurs people for race, ethnicity, nationality, religion, gender, sexual orientation, gender identity or disability",
            "no_hate": "no attack on a protected group; includes quoting or condemning hate and neutral or friendly mentions of a group",
        },
    },
    "sexual_harassment": {
        "type": "choice",
        "instructions": "Is `message` sexual harassment?",
        "criteria": {
            "harassment": "unwanted sexual comments, requests or advances aimed at a person",
            "no_harassment": "no sexual harassment",
        },
    },
    "profanity": {
        "type": "choice",
        "instructions": "Does `message` contain swearing or vulgar language?",
        "criteria": {
            "profanity": "contains swear words or vulgar language, aimed at anyone or at no one",
            "no_profanity": "no swear words or vulgar language",
        },
    },
    "target": {
        "type": "choice",
        "instructions": "Who does `message` attack?",
        "criteria": {
            "player": "a specific person in the conversation",
            "group": "a group of people",
            "nobody": "nobody; the message attacks no one",
        },
    },
}
POSITIVE = {"insult": "insult", "threat": "threat", "identity_hate": "hate", "sexual_harassment": "harassment", "profanity": "profanity"}

TAXONOMY_NOTES = (
    "Laya sin ajustar (zero-shot) con seis preguntas redactadas para este benchmark y sin pulir: el "
    "resultado depende tanto de la redacción como del modelo. El inglés que Laya identifica va al "
    "checkpoint inglés (ModernBERT-large) y el resto al multilingüe (mmBERT-base). Hace una fila de "
    "encoder por pregunta: seis por mensaje."
)
APP_NOTES = (
    "La pregunta `insult` que usa hoy la app (insultos, slurs y amenazas en una sola pregunta) como flag. "
    "Es la línea de base: lo que ya tenemos."
)


def fingerprint(questions: dict) -> str:
    return hashlib.sha1(json.dumps(questions, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def flatten(result: dict) -> dict[str, float]:
    raw = {f"{qid}.{option}": float(p) for qid, answer in result["answers"].items() for option, p in answer["probabilities"].items()}
    raw["routed_english"] = 1.0 if result["routing"]["model"] == "english" else 0.0
    return raw


def taxonomy_scores(raw: dict[str, float]) -> dict[str, float]:
    scores = {category: raw[f"{category}.{option}"] for category, option in POSITIVE.items()}
    scores[TARGETS_PLAYER] = raw["target.player"]
    return scores


def app_scores(raw: dict[str, float]) -> dict[str, float]:
    return {FLAG: raw["insult.insult"]}


class LayaContestant:
    languages = frozenset(LANGS)
    license = "Apache-2.0"

    def __init__(self, name: str, questions: dict, scores, notes: str, device: str | None = None, router=None):
        self.name = name
        self.questions = questions
        self._scores = scores
        self.notes = notes
        self.device = device
        self._router = router
        # The questions, the laya version and the routing setup all change the raw answers.
        self.version = f"q{fingerprint(questions)}-laya{metadata.version('laya')}-default-multilingual"

    def _get_router(self):
        if self._router is None:
            self._router = build_router(device=self.device)
            self._router.preload(["english", "multilingual"])
        return self._router

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        requests = [{"state": {STATE_KEY: text}, "questions": self.questions} for text in texts]
        return [flatten(result) for result in self._get_router().predict_batch(requests)]

    def to_scores(self, raw: dict[str, float]) -> dict[str, float]:
        return self._scores(raw)

    def unload(self) -> None:
        self._router = None
        free_gpu()


def laya_taxonomy(device: str | None = None, router=None) -> LayaContestant:
    return LayaContestant("laya", TAXONOMY_QUESTIONS, taxonomy_scores, TAXONOMY_NOTES, device=device, router=router)


def laya_app(device: str | None = None, router=None) -> LayaContestant:
    return LayaContestant("laya-app", APP_QUESTIONS, app_scores, APP_NOTES, device=device, router=router)
```

- [ ] **Step 5: Run the fast tests.**

Run: `uv run pytest tests/test_bench_laya.py tests/test_routing.py tests/test_classifier.py -v`
Expected: all pass.

- [ ] **Step 6: Add the slow test with the real Laya.** Append to `tests/test_bench_slow.py`:

```python
def test_real_laya_answers_every_taxonomy_question():
    from bench.contestants.base import effective_scores
    from bench.contestants.laya import laya_taxonomy

    contestant = laya_taxonomy()
    scores = [effective_scores(contestant, r) for r in contestant.predict(["sos un pelotudo", "buena mano, gg"], "es")]

    assert all(0.0 <= s["flag"] <= 1.0 and 0.0 <= s["targets_player"] <= 1.0 for s in scores)
```

Run: `uv run pytest -m slow tests/test_bench_slow.py -k laya -v`
Expected: PASS. If Laya raises `ValueError` about option markers past `max_len`, report BLOCKED with the message; the question wording is too long.

- [ ] **Step 7: Commit.** Summary: "add Laya contestants and a device option on build_router".

```bash
git add app/classifier.py bench/contestants/laya.py tests/test_bench_laya.py tests/test_routing.py tests/test_bench_slow.py
git commit
```

---

### Task 11: Detoxify and Horizon (HF) contestants, and the registry

**Files:**
- Create: `bench/contestants/detoxify.py`, `bench/contestants/hf.py`
- Modify: `bench/contestants/__init__.py`
- Test: `tests/test_bench_contestants.py`; append to `tests/test_bench_slow.py`

**Interfaces:**
- Consumes: `bench.contestants.base.map_labels`, `bench.taxonomy.LANGS`, and the Laya factories from Task 10.
- Produces:
  - `DETOXIFY_STYLE`
  - `DetoxifyContestant(device=None, model=None)`, with `unload()`; `HFContestant` also has `unload()`
  - `HFContestant(name, model_id, revision, label_map, notes, license, languages=..., max_length=256, device=None, loader=None)`
  - `HORIZON_REVISION`, `horizon_mmbert(device=None)`
  - `FACTORIES: dict[str, Callable[[str | None], Contestant]]`
  - `DEFAULT_CONTESTANTS = ("laya", "laya-app", "detoxify", "horizon-mmbert")`
  - `OPEN_LICENSES = ("Apache-2.0", "MIT")`
  - `build_contestants(names, device=None, factories=None) -> list`

- [ ] **Step 1: Confirm the pinned Horizon revision.** The plan pins `dbf12a9915275078e580707bc2396b07585da31e`, the revision Task 1 loaded. Keep it even if the publisher has pushed newer commits: a new revision is a new, unchecked model.

- [ ] **Step 2: Write the failing tests.** Create `tests/test_bench_contestants.py`:

```python
import re
from types import SimpleNamespace

import pytest
import torch

from bench.contestants import DEFAULT_CONTESTANTS, FACTORIES, OPEN_LICENSES, build_contestants
from bench.contestants.base import effective_scores
from bench.contestants.detoxify import DETOXIFY_STYLE, DetoxifyContestant
from bench.contestants.hf import HORIZON_REVISION, HFContestant
from bench.taxonomy import FLAG

DETOX_OUT = {"toxicity": [0.99, 0.1], "severe_toxicity": [0.5, 0.0], "obscene": [0.9, 0.0], "identity_attack": [0.2, 0.0],
             "insult": [0.7, 0.05], "threat": [0.1, 0.0], "sexual_explicit": [0.3, 0.0]}


class FakeDetoxify:
    def __init__(self, out):
        self.out = out

    def predict(self, texts):
        return self.out


def test_detoxify_returns_one_raw_dict_per_text_and_flag_ignores_toxicity():
    contestant = DetoxifyContestant(model=FakeDetoxify(DETOX_OUT))

    raw = contestant.predict(["sos un pelotudo", "buena mano"], "es")

    assert raw[0]["toxicity"] == 0.99 and raw[1]["insult"] == 0.05
    assert effective_scores(contestant, raw[0]) == {
        "insult": 0.7, "threat": 0.1, "identity_hate": 0.2, "sexual_harassment": 0.3, "profanity": 0.9, FLAG: 0.7,
    }


def test_detoxify_handles_scalar_outputs_for_a_single_text():
    contestant = DetoxifyContestant(model=FakeDetoxify({k: v[0] for k, v in DETOX_OUT.items()}))

    assert contestant.predict(["x"], "es")[0]["insult"] == 0.7


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, texts, **kwargs):
        self.kwargs = kwargs
        n = len(texts)
        return {"input_ids": torch.zeros((n, 4), dtype=torch.long), "attention_mask": torch.ones((n, 4), dtype=torch.long)}


class FakeModel:
    config = SimpleNamespace(id2label={0: "toxicity", 1: "insult", 2: "identity_attack"})

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0, -2.0]] * input_ids.shape[0]))


def test_hf_contestant_applies_sigmoid_names_outputs_and_truncates():
    tokenizer = FakeTokenizer()
    contestant = HFContestant("h", "org/model", "a" * 40, DETOXIFY_STYLE, "", "MIT", device="cpu",
                              loader=lambda: (tokenizer, FakeModel()))

    [raw] = contestant.predict(["hola"], "es")

    assert raw["toxicity"] == pytest.approx(0.5)
    assert raw["insult"] == pytest.approx(0.8808, abs=1e-4)
    assert tokenizer.kwargs["truncation"] is True and tokenizer.kwargs["max_length"] == 256
    assert contestant.to_scores(raw) == {"insult": raw["insult"], "identity_hate": raw["identity_attack"]}
    assert contestant.version == "a" * 12


def test_unload_releases_the_model_and_the_next_prediction_reloads_it():
    loads = []

    def loader():
        loads.append(1)
        return FakeTokenizer(), FakeModel()

    contestant = HFContestant("h", "org/model", "a" * 40, DETOXIFY_STYLE, "", "MIT", device="cpu", loader=loader)
    contestant.predict(["a"], "es")
    contestant.unload()
    contestant.predict(["b"], "es")

    assert len(loads) == 2
    detoxify = DetoxifyContestant(model=FakeDetoxify(DETOX_OUT))
    detoxify.unload()
    assert detoxify._model is None


def test_horizon_is_pinned_to_a_full_revision_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", HORIZON_REVISION)


def test_registry_builds_every_contestant_without_loading_a_model():
    contestants = build_contestants(list(FACTORIES))

    assert [c.name for c in contestants] == list(FACTORIES)
    assert all(c.version and c.notes and c.languages for c in contestants)
    assert set(DEFAULT_CONTESTANTS) == set(FACTORIES)


def test_every_contestant_is_open_source():
    # The team only uses open, self-hostable models: no paid or hosted APIs.
    assert {c.name: c.license for c in build_contestants(list(FACTORIES)) if c.license not in OPEN_LICENSES} == {}


def test_unknown_contestant_is_an_error():
    with pytest.raises(ValueError, match="desconocido"):
        build_contestants(["gpt-9"])
```

- [ ] **Step 3: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_contestants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.contestants.detoxify'`.

- [ ] **Step 4: Implement.** Create `bench/contestants/detoxify.py`:

```python
"""Detoxify's multilingual model: XLM-R trained on Jigsaw 2020, seven outputs."""

from __future__ import annotations

from importlib import metadata

from bench.contestants.base import free_gpu, map_labels
from bench.taxonomy import LANGS

# Detoxify-style label names → taxonomy. `toxicity` and `severe_toxicity` stay out: they include profanity.
DETOXIFY_STYLE = {
    "insult": "insult",
    "threat": "threat",
    "identity_attack": "identity_hate",
    "sexual_explicit": "sexual_harassment",
    "obscene": "profanity",
}


class DetoxifyContestant:
    name = "detoxify"
    languages = frozenset(LANGS)
    license = "Apache-2.0"
    notes = (
        "XLM-R entrenado con Jigsaw 2020. Fuera de `toxicity`, sus categorías se aprendieron de etiquetas en "
        "inglés traducidas, así que en español y portugués son débiles. `sexual_explicit` detecta contenido "
        "sexual, no acoso."
    )

    def __init__(self, device: str | None = None, model=None):
        self.device = device
        self._model = model
        self.version = f"multilingual-{metadata.version('detoxify')}"

    def _get_model(self):
        if self._model is None:
            import torch
            from detoxify import Detoxify

            self._model = Detoxify("multilingual", device=self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return self._model

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        out = self._get_model().predict(texts)
        # A one-text batch can come back as scalars instead of one-item lists.
        columns = {label: list(values) if hasattr(values, "__len__") else [values] for label, values in out.items()}
        return [{label: float(values[i]) for label, values in columns.items()} for i in range(len(texts))]

    def to_scores(self, raw: dict[str, float]) -> dict[str, float]:
        return map_labels(raw, DETOXIFY_STYLE)

    def unload(self) -> None:
        self._model = None
        free_gpu()
```

Create `bench/contestants/hf.py`, with `HORIZON_REVISION` set to the SHA from Step 1:

```python
"""Any Hugging Face multi-label classifier with sigmoid outputs, loaded at a pinned revision."""

from __future__ import annotations

from bench.contestants.base import free_gpu, map_labels
from bench.contestants.detoxify import DETOXIFY_STYLE
from bench.taxonomy import LANGS

HORIZON_REVISION = "dbf12a9915275078e580707bc2396b07585da31e"  # checked on 2026-10-06 (Task 1)
HORIZON_NOTES = (
    "mmBERT-base ajustado con Civil Comments traducido por un LLM. Lo publicó una cuenta creada en "
    "septiembre de 2026, sin trayectoria, y no sabemos con qué datos exactos se entrenó: si vio ToLD-Br, "
    "OLID-BR o HateCheck, sus números en esas fuentes están inflados. Se carga solo en safetensors y sin "
    "código remoto."
)


class HFContestant:
    def __init__(self, name: str, model_id: str, revision: str, label_map: dict[str, str], notes: str, license: str,
                 languages=LANGS, max_length: int = 256, device: str | None = None, loader=None):
        self.name = name
        self.model_id = model_id
        self.revision = revision
        self.label_map = label_map
        self.notes = notes
        self.license = license
        self.languages = frozenset(languages)
        self.max_length = max_length
        self.device = device
        self.version = revision[:12]
        self._loader = loader or self._from_hub
        self._loaded = None

    def _from_hub(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        # Safetensors only and no remote code: the weights come from a publisher nobody has vetted.
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=self.revision, use_safetensors=True)
        return tokenizer, model

    def _get(self):
        if self._loaded is None:
            import torch

            tokenizer, model = self._loader()
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._loaded = (tokenizer, model.to(device).eval(), device)
        return self._loaded

    def predict(self, texts: list[str], lang: str) -> list[dict[str, float]]:
        import torch

        tokenizer, model, device = self._get()
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(**encoded).logits.float()).cpu().tolist()
        id2label = model.config.id2label
        return [{id2label[i]: p for i, p in enumerate(row)} for row in probabilities]

    def to_scores(self, raw: dict[str, float]) -> dict[str, float]:
        return map_labels(raw, self.label_map)

    def unload(self) -> None:
        self._loaded = None
        free_gpu()


def horizon_mmbert(device: str | None = None) -> HFContestant:
    return HFContestant("horizon-mmbert", "Horizon-Labs/multilingual-toxicity-base", HORIZON_REVISION,
                        DETOXIFY_STYLE, HORIZON_NOTES, "Apache-2.0", device=device)
```

Replace `bench/contestants/__init__.py` with:

```python
"""Contestants by name. Building one loads nothing; each model loads on its first prediction."""

from __future__ import annotations

from bench.contestants.base import Contestant, effective_scores, map_labels  # noqa: F401
from bench.contestants.detoxify import DetoxifyContestant
from bench.contestants.hf import horizon_mmbert
from bench.contestants.laya import laya_app, laya_taxonomy

FACTORIES = {
    "laya": lambda device: laya_taxonomy(device=device),
    "laya-app": lambda device: laya_app(device=device),
    "detoxify": lambda device: DetoxifyContestant(device=device),
    "horizon-mmbert": lambda device: horizon_mmbert(device=device),
}
DEFAULT_CONTESTANTS = tuple(FACTORIES)
# Only open, self-hostable models: no paid or hosted APIs, and no message leaves our machines.
OPEN_LICENSES = ("Apache-2.0", "MIT")


def build_contestants(names, device: str | None = None, factories: dict | None = None) -> list:
    factories = FACTORIES if factories is None else factories
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"modelo desconocido: {', '.join(unknown)}. Opciones: {', '.join(factories)}")
    return [factories[n](device) for n in names]
```

- [ ] **Step 5: Run the tests again.**

Run: `uv run pytest tests/test_bench_contestants.py tests/test_bench_laya.py -v`
Expected: all pass.

- [ ] **Step 6: Add the slow test with the real classifiers.** Append to `tests/test_bench_slow.py`:

```python
@pytest.mark.parametrize("name", ["detoxify", "horizon-mmbert"])
def test_real_classifiers_score_a_spanish_insult_above_a_greeting(name):
    from bench.contestants import build_contestants
    from bench.contestants.base import effective_scores

    [contestant] = build_contestants([name])
    insult, greeting = (effective_scores(contestant, r)["flag"]
                        for r in contestant.predict(["sos un pelotudo de mierda", "hola, suerte a todos"], "es"))

    assert insult > greeting
```

Run: `uv run pytest -m slow tests/test_bench_slow.py -k classifiers -v`
Expected: 2 passed.

- [ ] **Step 7: Commit.** Summary: "add Detoxify and Horizon mmBERT contestants".

```bash
git add bench/contestants/ tests/test_bench_contestants.py tests/test_bench_slow.py
git commit
```

### Task 12: Scoreboard

**Files:**
- Create: `bench/scoreboard.py`
- Test: `tests/test_bench_scoreboard.py`

**Interfaces:**
- Consumes:
  - `effective_scores` (Task 9)
  - `auroc`, `paired_auroc_diff`, `slice_metrics`, `threshold_at_fpr` (Task 3)
  - `split_of` (Task 8)
  - `gold`, `FLAG`, `SCORED` (Task 2)
- Produces:
  - `MIN_PER_CLASS = 30`
  - `slice_keys(example) -> list[str]`
  - `functionality_label(source) -> str`
  - `thresholds(...)`, `coverage(...)`, `slices(...)`, `comparisons(...)`, `functionalities(...)`, `routing(...)`
  - `build_scoreboard(examples, contestants, raw, n_boot=1000, seed=0, split=split_of) -> dict`, with keys `coverage`, `thresholds`, `slices`, `comparisons`, `functionalities` and `routing`, whose row shapes Task 13 renders:
    - coverage rows: `{source, lang, label, positives, negatives}`
    - slice rows: `{contestant, source, lang, label, n, positives, prevalence, auroc, auroc_ci, ap, threshold, recall_at_op, fpr_at_op, precision_at_op, enough}`
    - comparison rows: `{source, best, contestant, diff, lo, hi}`
    - functionality rows: `{contestant, source, functionality, n, accuracy}`
    - routing rows: `{contestant, source, english_share}`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench_scoreboard.py`:

```python
from bench.scoreboard import build_scoreboard, functionality_label, slice_keys
from bench.taxonomy import FLAG, Example


def split(example):
    return "calib" if example.id.split(":")[1].startswith("c") else "eval"


def ex(i, insult, source="suite-es", author=None, functionality=None):
    return Example(id=f"{source}:{i}", source=source, lang="es", text=f"t{i}",
                   labels={"insult": insult, "threat": False, "identity_hate": False},
                   author=author, functionality=functionality)


class Perfect:
    name = "perfect"
    version = "1"
    notes = ""
    languages = frozenset({"es"})

    def to_scores(self, raw):
        return {"insult": raw["p"], "threat": 0.0, "identity_hate": 0.0}


class Coin(Perfect):
    name = "coin"


class OnlyHate(Perfect):
    name = "only-hate"

    def to_scores(self, raw):
        return {"identity_hate": raw["p"]}


def dataset():
    """80 eval examples (half insults, half by human authors) and 20 calibration ones."""
    examples, perfect, coin = [], {}, {}
    for i in range(80):
        e = ex(f"e{i}", i % 2 == 0, author="human" if i % 4 < 2 else "generated", functionality="insult_direct")
        examples.append(e)
        perfect[e.id], coin[e.id] = {"p": 0.9 if i % 2 == 0 else 0.05}, {"p": 0.5}
    for i in range(20):
        e = ex(f"c{i}", i % 2 == 0)
        examples.append(e)
        perfect[e.id], coin[e.id] = {"p": 0.9 if i % 2 == 0 else 0.01 * i}, {"p": 0.5}
    return examples, {"perfect": perfect, "coin": coin, "only-hate": perfect}


def board():
    examples, raw = dataset()
    return build_scoreboard(examples, [Perfect(), Coin(), OnlyHate()], raw, n_boot=20, split=split)


def row(b, contestant, source, label):
    return next((r for r in b["slices"] if (r["contestant"], r["source"], r["label"]) == (contestant, source, label)), None)


def test_thresholds_come_from_the_calibration_split_only():
    # Calibration negatives score 0.01..0.19; with the 40 eval negatives (0.05) mixed in it would be 0.15.
    assert board()["thresholds"]["perfect"]["insult"] == 0.19


def test_a_perfect_contestant_gets_auroc_one_and_full_recall_at_its_operating_point():
    r = row(board(), "perfect", "suite-es", FLAG)

    assert (r["n"], r["positives"], r["auroc"], r["enough"]) == (80, 40, 1.0, True)
    assert (r["recall_at_op"], r["fpr_at_op"]) == (1.0, 0.0)


def test_suite_rows_are_also_reported_by_author_and_small_slices_are_marked():
    b = board()

    human = row(b, "perfect", "suite-es/human", FLAG)
    assert (human["n"], human["enough"]) == (40, False)
    assert row(b, "perfect", "suite-es/generated", FLAG) is not None


def test_a_single_class_label_has_no_auroc():
    r = row(board(), "perfect", "suite-es", "threat")

    assert r["positives"] == 0 and r["auroc"] is None and r["enough"] is False


def test_a_contestant_without_every_core_category_gets_no_flag_rows():
    b = board()

    assert row(b, "only-hate", "suite-es", FLAG) is None
    assert row(b, "only-hate", "suite-es", "identity_hate") is not None


def test_coverage_counts_the_eval_split():
    cov = {(c["source"], c["label"]): c for c in board()["coverage"]}

    assert (cov[("suite-es", "insult")]["positives"], cov[("suite-es", "insult")]["negatives"]) == (40, 40)
    assert cov[("suite-es/human", FLAG)]["positives"] == 20


def test_comparisons_measure_each_contestant_against_the_best():
    [c] = [c for c in board()["comparisons"] if c["source"] == "suite-es"]

    assert (c["best"], c["contestant"], c["diff"]) == ("perfect", "coin", -0.5)
    assert c["hi"] < 0


def test_functionality_accuracy_uses_the_operating_threshold():
    rows = {(r["contestant"], r["functionality"]): r for r in board()["functionalities"]}

    assert rows[("perfect", "insult_direct")] == {"contestant": "perfect", "source": "suite-es",
                                                  "functionality": "insult_direct", "n": 80, "accuracy": 1.0}
    assert functionality_label("hatecheck-pt") == "identity_hate" and functionality_label("suite-es") == FLAG


def test_routing_reports_the_share_sent_to_english():
    examples = [ex("e1", True), ex("e2", False)]
    raw = {"perfect": {"suite-es:e1": {"p": 0.9, "routed_english": 1.0}, "suite-es:e2": {"p": 0.1, "routed_english": 0.0}}}

    b = build_scoreboard(examples, [Perfect()], raw, n_boot=5, split=split)

    assert b["routing"] == [{"contestant": "perfect", "source": "suite-es", "english_share": 0.5}]


def test_slice_keys_add_the_author_when_there_is_one():
    assert slice_keys(ex("e1", True, author="human")) == ["suite-es", "suite-es/human"]
    assert slice_keys(ex("e1", True)) == ["suite-es"]
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_scoreboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.scoreboard'`.

- [ ] **Step 3: Implement.** Create `bench/scoreboard.py`:

```python
"""From raw predictions to the numbers in the report.

Each contestant gets one operating threshold per label, chosen on the calibration split where 5% of the
negatives get flagged. Everything reported is measured on the eval split, per source: pooling sources
would mostly measure which dataset a text came from.
"""

from __future__ import annotations

from collections import defaultdict

from bench.contestants.base import effective_scores
from bench.metrics import auroc, paired_auroc_diff, slice_metrics, threshold_at_fpr
from bench.sources import split_of
from bench.taxonomy import FLAG, SCORED, gold

MIN_PER_CLASS = 30


def slice_keys(example) -> list[str]:
    """The source, plus source/author for the suite, so generated and human-written rows show apart."""
    keys = [example.source]
    if example.author:
        keys.append(f"{example.source}/{example.author}")
    return keys


def functionality_label(source: str) -> str:
    # HateCheck measures hate speech; the suite's functionalities are about moderation as a whole.
    return "identity_hate" if source.startswith("hatecheck") else FLAG


def _pairs(examples, scores: dict, label: str):
    """(example, gold, score) for every example the contestant scored and that has gold for `label`."""
    for e in examples:
        value, expected = scores.get(e.id, {}).get(label), gold(e, label)
        if value is not None and expected is not None:
            yield e, expected, value


def thresholds(examples, scores_by_contestant, split=split_of) -> dict[str, dict[str, float | None]]:
    calib = [e for e in examples if split(e) == "calib"]
    out = {}
    for name, scores in scores_by_contestant.items():
        out[name] = {}
        for label in SCORED:
            pairs = list(_pairs(calib, scores, label))
            out[name][label] = threshold_at_fpr([g for _, g, _ in pairs], [s for _, _, s in pairs]) if pairs else None
    return out


def coverage(examples, split=split_of) -> list[dict]:
    counts = defaultdict(lambda: [0, 0])
    for e in examples:
        if split(e) != "eval":
            continue
        for label in SCORED:
            expected = gold(e, label)
            if expected is None:
                continue
            for key in slice_keys(e):
                counts[(key, e.lang, label)][0 if expected else 1] += 1
    return [{"source": s, "lang": lang, "label": label, "positives": p, "negatives": n}
            for (s, lang, label), (p, n) in sorted(counts.items())]


def slices(examples, scores_by_contestant, thr, n_boot=1000, seed=0, split=split_of) -> list[dict]:
    evaluated = [e for e in examples if split(e) == "eval"]
    rows = []
    for name, scores in scores_by_contestant.items():
        groups = defaultdict(lambda: ([], [], []))
        for label in SCORED:
            for e, expected, value in _pairs(evaluated, scores, label):
                for key in slice_keys(e):
                    golds, values, clusters = groups[(key, e.lang, label)]
                    golds.append(expected)
                    values.append(value)
                    clusters.append(e.cluster_key)
        for (key, lang, label), (golds, values, clusters) in sorted(groups.items()):
            m = slice_metrics(golds, values, clusters, thr[name][label], n_boot=n_boot, seed=seed)
            m["enough"] = m["positives"] >= MIN_PER_CLASS and m["n"] - m["positives"] >= MIN_PER_CLASS
            rows.append({"contestant": name, "source": key, "lang": lang, "label": label, **m})
    return rows


def comparisons(examples, scores_by_contestant, n_boot=1000, seed=0, split=split_of) -> list[dict]:
    """Per source, `flag` AUROC of each contestant minus the best one's, on the items all of them scored."""
    by_source = defaultdict(list)
    for e in examples:
        if split(e) == "eval" and gold(e, FLAG) is not None:
            for key in slice_keys(e):
                by_source[key].append(e)
    rows = []
    for key, items in sorted(by_source.items()):
        names = [n for n, s in scores_by_contestant.items() if all(FLAG in s.get(e.id, {}) for e in items)]
        golds = [gold(e, FLAG) for e in items]
        values = {n: [scores_by_contestant[n][e.id][FLAG] for e in items] for n in names}
        aurocs = {n: auroc(golds, values[n]) for n in names}
        names = [n for n in names if aurocs[n] is not None]
        if len(names) < 2:
            continue
        best = max(names, key=lambda n: aurocs[n])
        clusters = [e.cluster_key for e in items]
        for n in names:
            if n != best:
                diff = paired_auroc_diff(golds, values[n], values[best], clusters, n_boot=n_boot, seed=seed)
                rows.append({"source": key, "best": best, "contestant": n, **diff})
    return rows


def functionalities(examples, scores_by_contestant, thr, split=split_of) -> list[dict]:
    """Share of cases per functionality on the right side of the contestant's operating threshold."""
    hits = defaultdict(list)
    evaluated = [e for e in examples if split(e) == "eval" and e.functionality]
    for name, scores in scores_by_contestant.items():
        for e in evaluated:
            label = functionality_label(e.source)
            threshold, value, expected = thr[name].get(label), scores.get(e.id, {}).get(label), gold(e, label)
            if threshold is not None and value is not None and expected is not None:
                hits[(name, e.source, e.functionality)].append((value > threshold) == expected)
    return [{"contestant": n, "source": s, "functionality": f, "n": len(h), "accuracy": sum(h) / len(h)}
            for (n, s, f), h in sorted(hits.items())]


def routing(examples, raw) -> list[dict]:
    """For Laya contestants: the share of each source's messages its router sent to the English checkpoint."""
    rows = []
    for name, predictions in raw.items():
        by_source = defaultdict(list)
        for e in examples:
            r = predictions.get(e.id)
            if r is not None and "routed_english" in r:
                by_source[e.source].append(r["routed_english"])
        rows += [{"contestant": name, "source": s, "english_share": sum(v) / len(v)} for s, v in sorted(by_source.items())]
    return rows


def build_scoreboard(examples, contestants, raw, n_boot: int = 1000, seed: int = 0, split=split_of) -> dict:
    scores = {c.name: {i: effective_scores(c, r) for i, r in raw.get(c.name, {}).items()} for c in contestants}
    thr = thresholds(examples, scores, split)
    return {
        "coverage": coverage(examples, split),
        "thresholds": thr,
        "slices": slices(examples, scores, thr, n_boot, seed, split),
        "comparisons": comparisons(examples, scores, n_boot, seed, split),
        "functionalities": functionalities(examples, scores, thr, split),
        "routing": routing(examples, raw),
    }
```

- [ ] **Step 4: Run the tests again.**

Run: `uv run pytest tests/test_bench_scoreboard.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.** Summary: "add scoreboard: thresholds, slices, comparisons, functionalities".

```bash
git add bench/scoreboard.py tests/test_bench_scoreboard.py
git commit
```

---

### Task 13: Report, latency and CLI

**Files:**
- Create: `bench/report.py`, `bench/latency.py`, `bench/cli.py`, `scripts/bench.py`
- Test: `tests/test_bench_report.py`, `tests/test_bench_cli.py`

**Interfaces:**
- Consumes:
  - the `build_scoreboard` output shape (Task 12)
  - `run`, `PredictionCache` (Task 9)
  - `FACTORIES`, `DEFAULT_CONTESTANTS`, `build_contestants` (Task 11)
  - `SOURCES`, `load_sources`, `stable_hash`, `DEFAULT_MAX_PER_SOURCE` (Task 8)
  - `parse_suite`, `agreement` (Task 4)
  - `app.jsonfile.read_json`, `write_json_atomic`
- Produces:
  - `render(board) -> str`
  - `percentile(values, q) -> float`
  - `measure(contestant, texts, lang, batch_size=32, clock=time.perf_counter) -> {n, p50_ms, p95_ms, throughput_per_s, batch_size}`
  - `main(argv=None, sources=None, factories=None) -> int`, with subcommands `sources`, `run`, `latency` and `agreement`
  - `board["meta"]` = `{created_at, preliminary, n_boot, contestants: [{name, version, notes}], sources: [{name, lang, license, n}]}`
  - `board["latency"]` = a list of `{contestant, device, lang, n, p50_ms, p95_ms, throughput_per_s, batch_size}`

- [ ] **Step 1: Write the failing report tests.** Create `tests/test_bench_report.py`:

```python
from bench.report import render

FLAG_ROW = {"contestant": "laya", "source": "suite-es", "lang": "es", "label": "flag", "n": 90, "positives": 40,
            "prevalence": 0.44, "auroc": 0.91, "auroc_ci": [0.85, 0.95], "ap": 0.9, "threshold": 0.42,
            "recall_at_op": 0.7, "fpr_at_op": 0.06, "precision_at_op": 0.9, "enough": True}


def board(**overrides):
    b = {
        "meta": {"created_at": "2026-10-06T12:00:00Z", "preliminary": False, "n_boot": 100,
                 "contestants": [{"name": "laya", "version": "q1", "license": "Apache-2.0", "notes": "Nota de laya."},
                                 {"name": "detoxify", "version": "m", "license": "Apache-2.0", "notes": "Nota de detoxify."}],
                 "sources": [{"name": "suite-es", "lang": "es", "license": "propia", "n": 100}]},
        "coverage": [{"source": "suite-es", "lang": "es", "label": "flag", "positives": 40, "negatives": 50},
                     {"source": "suite-es", "lang": "es", "label": "profanity", "positives": 10, "negatives": 5}],
        "thresholds": {"laya": {"flag": 0.42, "insult": None}, "detoxify": {"flag": None}},
        "slices": [FLAG_ROW, {**FLAG_ROW, "label": "profanity", "n": 15, "positives": 10, "enough": False}],
        "comparisons": [{"source": "suite-es", "best": "laya", "contestant": "detoxify", "diff": -0.1, "lo": -0.2, "hi": -0.01}],
        "functionalities": [{"contestant": "laya", "source": "suite-es", "functionality": "negation", "n": 30, "accuracy": 0.9}],
        "routing": [{"contestant": "laya", "source": "suite-es", "english_share": 0.02}],
        "latency": [],
    }
    return b | overrides


def test_flag_cell_shows_auroc_interval_and_operating_point():
    assert "0.91 [0.85–0.95]<br>R 70% · FPR 6%" in render(board())


def test_missing_contestant_rows_say_na_and_uncovered_labels_say_dash():
    md = render(board())
    flag_line = next(line for line in md.splitlines() if line.startswith("| detoxify |"))

    assert "n/a" in flag_line
    assert "| laya | — |" in md  # per-category table: insult has no coverage, so it is a dash


def test_small_slices_say_pocos_datos_instead_of_numbers():
    assert "pocos datos (10+/5−)" in render(board())


def test_preliminary_banner_only_when_preliminary():
    assert "PRELIMINAR" not in render(board())
    assert "PRELIMINAR" in render(board(meta={**board()["meta"], "preliminary": True}))


def test_comparisons_functionalities_routing_notes_and_thresholds_are_rendered():
    md = render(board())

    assert "| suite-es | laya | detoxify | -0.10 [-0.20 – -0.01] |" in md
    assert "| negation | 30 | 90% | n/a |" in md
    assert "| laya | suite-es | 2% |" in md
    assert "Nota de laya. Umbrales de operación: flag=0.42." in md
    assert "Nota de detoxify. Umbrales de operación: n/a." in md


def test_latency_section_only_when_measured():
    assert "## Latencia" not in render(board())
    latency = [{"contestant": "laya", "device": "cpu", "lang": "es", "n": 200, "p50_ms": 120.4, "p95_ms": 300.2,
                "throughput_per_s": 15.2, "batch_size": 32}]
    assert "| laya | cpu | es | 120 | 300 | 15 |" in render(board(latency=latency))
```

- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.report'`.

- [ ] **Step 3: Implement the report.** Create `bench/report.py`:

```python
"""bench_out/scoreboard.md, in Spanish like the rest of the team-facing docs."""

from __future__ import annotations

from collections import defaultdict

from bench.taxonomy import FLAG, SCORED

LEGEND = [
    "## Cómo leer esto",
    "",
    "- **AUROC [IC 95%]**: probabilidad de que el modelo puntúe más alto un mensaje dañino que uno sano. "
    "0.50 es azar y 1.00 es perfecto. El intervalo muestra cuánto puede variar por azar.",
    "- **R · FPR**: con el umbral de operación de cada modelo (el que marca al 5% de los mensajes sanos de la "
    "partición de calibración), qué parte de los dañinos detecta (R) y qué parte de los sanos marca (FPR) en esa fuente.",
    "- **pocos datos (a+/b−)**: menos de 30 ejemplos de alguna clase; no sacar conclusiones.",
    "- **n/a**: el modelo no responde esa categoría. **—**: la fuente no tiene etiquetas de esa categoría.",
    "- Estas fuentes tienen muchos más mensajes dañinos que un chat real (donde son el 1–5%): en producción "
    "la precisión va a ser menor.",
    "",
]


def _num(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(r) + " |" for r in rows)]


def _auroc(row: dict | None) -> str:
    if row is None:
        return "n/a"
    if not row["enough"]:
        return f"pocos datos ({row['positives']}+/{row['n'] - row['positives']}−)"
    ci = row["auroc_ci"]
    return _num(row["auroc"]) + (f" [{_num(ci[0])}–{_num(ci[1])}]" if ci else "")


def _flag(row: dict | None) -> str:
    cell = _auroc(row)
    if row is not None and row["enough"] and row["recall_at_op"] is not None:
        cell += f"<br>R {_pct(row['recall_at_op'])} · FPR {_pct(row['fpr_at_op'])}"
    return cell


def render(board: dict) -> str:
    meta = board["meta"]
    names = [c["name"] for c in meta["contestants"]]
    rows = {(r["contestant"], r["source"], r["label"]): r for r in board["slices"]}
    covered = {(r["source"], r["label"]): r for r in board["coverage"]}
    sources = sorted({(r["source"], r["lang"]) for r in board["coverage"]})

    def cell(name, source, label, fmt=_auroc):
        if (source, label) not in covered:
            return "—"
        return fmt(rows.get((name, source, label)))

    lines = ["# Scoreboard de moderación", ""]
    if meta["preliminary"]:
        lines += ["> **PRELIMINAR:** incluye filas de la suite que nadie revisó todavía. Sirve para probar la "
                  "herramienta, no para decidir.", ""]
    lines += [f"Generado: {meta['created_at']} · Remuestreos bootstrap: {meta['n_boot']}", ""] + LEGEND

    lines += ["## Cobertura (ejemplos de evaluación: dañinos/sanos)", ""]
    lines += _table(["fuente", "idioma", *SCORED], [
        [s, lang, *(f"{covered[(s, label)]['positives']}/{covered[(s, label)]['negatives']}" if (s, label) in covered else "—"
                    for label in SCORED)]
        for s, lang in sources
    ])

    lines += ["", "## ¿Hay que moderar? (`flag` = insult, threat o identity_hate)", ""]
    lines += _table(["modelo", *(s for s, _ in sources)], [[n, *(cell(n, s, FLAG, _flag) for s, _ in sources)] for n in names])

    if board["comparisons"]:
        lines += ["", "## Diferencia de AUROC contra el mejor de cada fuente (`flag`, bootstrap pareado)", "",
                  "Si el intervalo incluye 0, no hay evidencia de que el mejor sea mejor de verdad.", ""]
        lines += _table(["fuente", "mejor", "modelo", "diferencia [IC 95%]"], [
            [r["source"], r["best"], r["contestant"], f"{_num(r['diff'])} [{_num(r['lo'])} – {_num(r['hi'])}]"]
            for r in board["comparisons"]
        ])

    lines += ["", "## Por categoría (AUROC)", ""]
    for s, lang in sources:
        lines += [f"### {s} ({lang})", ""]
        lines += _table(["modelo", *SCORED], [[n, *(cell(n, s, label) for label in SCORED)] for n in names]) + [""]

    by_functionality = defaultdict(dict)
    for r in board["functionalities"]:
        by_functionality[(r["source"], r["functionality"])][r["contestant"]] = r
    if by_functionality:
        lines += ["## Por funcionalidad (aciertos con el umbral de operación)", "",
                  "En HateCheck se mide `identity_hate`, que es lo que HateCheck evalúa; en la suite propia, `flag`.", ""]
        for source in sorted({s for s, _ in by_functionality}):
            table = [
                [f, str(max(r["n"] for r in per.values())), *(_pct(per[n]["accuracy"]) if n in per else "n/a" for n in names)]
                for (s, f), per in sorted(by_functionality.items()) if s == source
            ]
            lines += [f"### {source}", ""] + _table(["funcionalidad", "n", *names], table) + [""]

    if board["routing"]:
        lines += ["## Laya: mensajes que su router mandó al checkpoint inglés", ""]
        lines += _table(["modelo", "fuente", "al inglés"],
                        [[r["contestant"], r["source"], _pct(r["english_share"])] for r in board["routing"]]) + [""]

    if board.get("latency"):
        lines += ["## Latencia", ""]
        lines += _table(["modelo", "dispositivo", "idioma", "p50 ms", "p95 ms", "mensajes/s en lotes"], [
            [r["contestant"], r["device"], r["lang"], f"{r['p50_ms']:.0f}", f"{r['p95_ms']:.0f}",
             "n/a" if r["throughput_per_s"] is None else f"{r['throughput_per_s']:.0f}"]
            for r in board["latency"]
        ]) + [""]

    lines += ["## Modelos", ""]
    for c in meta["contestants"]:
        shown = ", ".join(f"{label}={_num(t)}" for label, t in board["thresholds"].get(c["name"], {}).items() if t is not None)
        lines.append(f"- **{c['name']}** (`{c['version']}`, {c['license']}): {c['notes']} "
                     f"Umbrales de operación: {shown or 'n/a'}.")
    lines += ["", "## Fuentes", ""]
    lines += _table(["fuente", "idioma", "licencia", "ejemplos"],
                    [[s["name"], s["lang"], s["license"], str(s["n"])] for s in meta["sources"]])
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the report tests.**

Run: `uv run pytest tests/test_bench_report.py -v`
Expected: all pass.

- [ ] **Step 5: Write the failing CLI and latency tests.** Create `tests/test_bench_cli.py`:

```python
import json

import pytest

from bench.cli import main
from bench.latency import measure, percentile
from bench.sources import Source
from bench.taxonomy import Example


class Fake:
    name = "fake"
    version = "1"
    notes = "Modelo de prueba."
    license = "MIT"
    languages = frozenset({"es"})

    def __init__(self, device=None):
        self.calls = 0

    def predict(self, texts, lang):
        self.calls += 1
        return [{"p": 0.9 if "idiota" in t else 0.1} for t in texts]

    def to_scores(self, raw):
        return {"insult": raw["p"], "threat": 0.0, "identity_hate": 0.0}


def fake_sources():
    examples = [Example(id=f"f:{i}", source="fake-es", lang="es", text=f"sos un idiota {i}" if i % 2 else f"hola {i}",
                        labels={"insult": bool(i % 2), "threat": False, "identity_hate": False}) for i in range(400)]
    return {"fake-es": Source("fake-es", "es", "test", lambda _: examples)}


def run_main(*args, factories=None):
    return main(list(args), sources=fake_sources(), factories=factories or {"fake": Fake})


def test_run_writes_markdown_and_json_scoreboards(tmp_path):
    assert run_main("run", "--contestants", "fake", "--out", str(tmp_path), "--bootstrap", "20") == 0

    board = json.loads((tmp_path / "scoreboard.json").read_text(encoding="utf-8"))
    [flag] = [r for r in board["slices"] if r["label"] == "flag" and r["source"] == "fake-es"]
    assert flag["auroc"] == 1.0 and flag["enough"] is True
    md = (tmp_path / "scoreboard.md").read_text(encoding="utf-8")
    assert "Modelo de prueba." in md and "PRELIMINAR" not in md


def test_include_unreviewed_marks_the_scoreboard_preliminary(tmp_path):
    run_main("run", "--contestants", "fake", "--out", str(tmp_path), "--bootstrap", "5", "--include-unreviewed")

    assert "PRELIMINAR" in (tmp_path / "scoreboard.md").read_text(encoding="utf-8")


def test_unknown_names_are_reported(tmp_path, capsys):
    assert run_main("run", "--sources", "nope", "--contestants", "fake", "--out", str(tmp_path)) == 1
    assert "nope" in capsys.readouterr().out


def test_latency_keeps_one_entry_per_contestant_device_and_language(tmp_path):
    args = ("latency", "--contestants", "fake", "--device", "cpu", "--lang", "es", "--n", "5", "--out", str(tmp_path))

    assert run_main(*args) == 0 and run_main(*args) == 0

    entries = json.loads((tmp_path / "latency.json").read_text(encoding="utf-8"))
    assert [(e["contestant"], e["device"], e["lang"], e["n"]) for e in entries] == [("fake", "cpu", "es", 5)]


def test_latency_unloads_each_contestant_after_measuring_it(tmp_path):
    unloaded = []

    class Unloading(Fake):
        def unload(self):
            unloaded.append(self.name)

    assert run_main("latency", "--contestants", "fake", "--n", "3", "--out", str(tmp_path), factories={"fake": Unloading}) == 0
    assert unloaded == ["fake"]


def test_percentile_uses_nearest_rank():
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(list(range(1, 11)), 95) == 10


def test_measure_times_single_messages_and_batches_after_a_warm_up():
    class Clock:
        t = 0.0

        def __call__(self):
            self.t += 0.01
            return self.t

    contestant = Fake()

    result = measure(contestant, [f"m{i}" for i in range(10)], "es", batch_size=4, clock=Clock())

    assert contestant.calls == 1 + 10 + 3
    assert result["p50_ms"] == pytest.approx(10.0) and result["p95_ms"] == pytest.approx(10.0)
    assert result["throughput_per_s"] == pytest.approx(10 / 0.01)
    assert (result["n"], result["batch_size"]) == (10, 4)
```

- [ ] **Step 6: Run them to see them fail.**

Run: `uv run pytest tests/test_bench_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.cli'`.

- [ ] **Step 7: Implement latency, the CLI and the script.** Create `bench/latency.py`:

```python
"""How long a contestant takes per message, one at a time and in batches."""

from __future__ import annotations

import math
import time


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def measure(contestant, texts: list[str], lang: str, batch_size: int = 32, clock=time.perf_counter) -> dict:
    contestant.predict(texts[:1], lang)  # loads the model; not timed
    per_message = []
    for text in texts:
        start = clock()
        contestant.predict([text], lang)
        per_message.append((clock() - start) * 1000)
    start = clock()
    for i in range(0, len(texts), batch_size):
        contestant.predict(texts[i:i + batch_size], lang)
    total = clock() - start
    return {
        "n": len(texts),
        "p50_ms": percentile(per_message, 50),
        "p95_ms": percentile(per_message, 95),
        "throughput_per_s": len(texts) / total if total > 0 else None,
        "batch_size": batch_size,
    }
```

Create `bench/cli.py`:

```python
"""Command line for the moderation benchmark (scripts/bench.py). See docs/benchmark.md."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from app.jsonfile import read_json, write_json_atomic
from bench.cache import PredictionCache
from bench.contestants import DEFAULT_CONTESTANTS, FACTORIES, build_contestants
from bench.latency import measure
from bench.report import render
from bench.runner import run
from bench.scoreboard import build_scoreboard
from bench.sources import DEFAULT_MAX_PER_SOURCE, SOURCES, load_sources, stable_hash
from bench.sources.suite import agreement, parse_suite
from bench.taxonomy import LANGS, SCORED, gold

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "bench_out"


class CliError(Exception):
    pass


def _pick(value: str | None, valid, default, kind: str) -> list[str]:
    names = [n.strip() for n in value.split(",") if n.strip()] if value else list(default)
    unknown = [n for n in names if n not in valid]
    if unknown:
        raise CliError(f"{kind} inexistente: {', '.join(unknown)}. Opciones: {', '.join(valid)}")
    return names


def cmd_sources(args, sources, factories) -> int:
    names = _pick(args.sources, sources, sources, "fuente")
    examples = load_sources(names, args.max_per_source, args.include_unreviewed, sources=sources)
    print(f"{'fuente':<14} {'idioma':<6} {'ejemplos':>8}  " + "  ".join(f"{label} (+/-)" for label in SCORED))
    for name in names:
        mine = [e for e in examples if e.source == name]
        counts = []
        for label in SCORED:
            values = [gold(e, label) for e in mine]
            counts.append(f"{values.count(True)}/{values.count(False)}")
        print(f"{name:<14} {sources[name].lang:<6} {len(mine):>8}  " + "  ".join(counts))
    return 0


def cmd_run(args, sources, factories) -> int:
    source_names = _pick(args.sources, sources, sources, "fuente")
    names = _pick(args.contestants, factories, DEFAULT_CONTESTANTS, "modelo")
    examples = load_sources(source_names, args.max_per_source, args.include_unreviewed, sources=sources)
    if not examples:
        raise CliError("No hay ejemplos para evaluar.")
    contestants = build_contestants(names, None if args.device == "auto" else args.device, factories=factories)
    out = Path(args.out)
    raw = run(contestants, examples, PredictionCache(out / "predictions"), batch_size=args.batch_size)
    print("Calculando métricas…")
    board = build_scoreboard(examples, contestants, raw, n_boot=args.bootstrap)
    board["meta"] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "preliminary": args.include_unreviewed,
        "n_boot": args.bootstrap,
        "contestants": [{"name": c.name, "version": c.version, "license": c.license, "notes": c.notes} for c in contestants],
        "sources": [{"name": n, "lang": sources[n].lang, "license": sources[n].license,
                     "n": sum(e.source == n for e in examples)} for n in source_names],
    }
    board["latency"] = read_json(out / "latency.json", [])
    write_json_atomic(out / "scoreboard.json", board)
    (out / "scoreboard.md").write_text(render(board), encoding="utf-8")
    print(f"Listo: {out / 'scoreboard.md'}")
    return 0


def cmd_latency(args, sources, factories) -> int:
    names = _pick(args.contestants, factories, DEFAULT_CONTESTANTS, "modelo")
    in_lang = [n for n, s in sources.items() if s.lang == args.lang]
    examples = load_sources(in_lang, include_unreviewed=True, sources=sources, log=lambda _: None)
    texts = [e.text for e in sorted(examples, key=lambda e: stable_hash(e.id))][: args.n]
    if not texts:
        raise CliError(f"No hay mensajes en '{args.lang}' para medir.")
    out = Path(args.out)
    entries = [e for e in read_json(out / "latency.json", [])
               if not (e["contestant"] in names and e["device"] == args.device and e["lang"] == args.lang)]
    for contestant in build_contestants(names, args.device, factories=factories):
        result = measure(contestant, texts, args.lang, batch_size=args.batch_size)
        entries.append({"contestant": contestant.name, "device": args.device, "lang": args.lang, **result})
        unload = getattr(contestant, "unload", None)
        if unload is not None:
            unload()
        print(f"{contestant.name:<18} p50 {result['p50_ms']:.0f} ms · p95 {result['p95_ms']:.0f} ms · "
              f"{result['throughput_per_s'] or 0:.0f} mensajes/s en lotes de {args.batch_size}")
    write_json_atomic(out / "latency.json", entries)
    return 0


def cmd_agreement(args, sources, factories) -> int:
    a, b = ([e for e, _ in parse_suite(Path(p).read_text(encoding="utf-8-sig"), "suite")] for p in (args.a, args.b))
    print(f"{'categoría':<20} {'filas':>6} {'kappa':>6}")
    for category, result in agreement(a, b).items():
        kappa = "n/a" if result["kappa"] is None else f"{result['kappa']:.2f}"
        print(f"{category:<20} {result['n']:>6} {kappa:>6}")
    return 0


def main(argv=None, sources=None, factories=None) -> int:
    sources = SOURCES if sources is None else sources
    factories = FACTORIES if factories is None else factories
    parser = argparse.ArgumentParser(prog="bench.py", description="Benchmark de moderación (ver docs/benchmark.md).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sources", help="cuántos ejemplos hay por fuente y categoría")
    p.add_argument("--sources", help="lista separada por comas (por defecto, todas)")
    p.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE)
    p.add_argument("--include-unreviewed", action="store_true")
    p.set_defaults(handler=cmd_sources)

    p = sub.add_parser("run", help="corre los modelos y escribe el scoreboard")
    p.add_argument("--sources", help="lista separada por comas (por defecto, todas)")
    p.add_argument("--contestants", help=f"lista separada por comas (por defecto: {','.join(DEFAULT_CONTESTANTS)})")
    p.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE)
    p.add_argument("--include-unreviewed", action="store_true", help="incluir filas sin revisar (scoreboard PRELIMINAR)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--bootstrap", type=int, default=1000, help="remuestreos para los intervalos de confianza")
    p.add_argument("--batch-size", type=int, default=32, help="mensajes por lote; bajalo si la GPU se queda sin memoria")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.set_defaults(handler=cmd_run)

    p = sub.add_parser("latency", help="mide cuánto tarda cada modelo por mensaje")
    p.add_argument("--contestants", help="lista separada por comas")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--lang", choices=list(LANGS), default="es")
    p.add_argument("--n", type=int, default=200, help="cantidad de mensajes")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.set_defaults(handler=cmd_latency)

    p = sub.add_parser("agreement", help="kappa entre dos revisiones de la suite")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(handler=cmd_agreement)

    args = parser.parse_args(argv)
    try:
        return args.handler(args, sources, factories)
    except (CliError, ValueError) as e:
        print(e)
        return 1
```

Create `scripts/bench.py`:

```python
"""Moderation benchmark. See docs/benchmark.md.

    uv run python scripts/bench.py sources
    uv run python scripts/bench.py run [--include-unreviewed] [--contestants laya,detoxify] [--device cuda]
    uv run python scripts/bench.py latency --device cpu
    uv run python scripts/bench.py agreement suites/es_casino.csv copia.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Run the fast suite.**

Run: `uv run pytest -v`
Expected: every test passes, old and new.

- [ ] **Step 9: Commit.** Summary: "add scoreboard report, latency measurement and bench CLI".

```bash
git add bench/report.py bench/latency.py bench/cli.py scripts/bench.py tests/test_bench_report.py tests/test_bench_cli.py
git commit
```

---

### Task 14: First preliminary run and docs

**Files:**
- Create: `docs/benchmarks/<run date>-preliminar.md`, copied from `bench_out/scoreboard.md`
- Modify: `docs/benchmark.md` (append), `README.md`

This task runs real models on the GPU (RTX 500 Ada Laptop, 4 GB). The first run downloads ~4 GB of models plus the datasets.

- [ ] **Step 1: Check the full test suite, including the slow tests.**

Run: `uv run pytest && uv run pytest -m slow`
Expected: everything passes.

- [ ] **Step 2: Inspect what the exam contains.**

Run: `uv run python scripts/bench.py sources --include-unreviewed`
Expected:
- suite-es ≈ 651 rows
- hatecheck-es/pt/en ≈ 3.5k–3.7k each
- toldbr 4,000
- olidbr 1,738
- `flag` has both positives and negatives in every source.

If a source has 0 negatives for `flag`, stop and report it as a mapping bug.

- [ ] **Step 3: Run the benchmark.**

Run: `uv run python scripts/bench.py run --include-unreviewed --device cuda`

If CUDA runs out of memory, rerun the same command with `--batch-size 8`; finished batches stay in the cache.

Expected:
- per-source progress lines for each contestant, ending with `Listo: …\bench_out\scoreboard.md`
- `scoreboard.md` starts with the PRELIMINAR banner, and its `flag` table has numbers for every contestant and every source.

- [ ] **Step 4: Measure latency.**

Run: `uv run python scripts/bench.py latency --device cpu --lang es --n 200`
Then: `uv run python scripts/bench.py latency --device cuda --lang es --n 200`
Expected: one line per contestant for each run. Laya should show several times the CPU latency of the single-pass classifiers, since it does six encoder rows per message.

- [ ] **Step 5: Re-render with latency and confirm the cache works.**

Run: `uv run python scripts/bench.py run --include-unreviewed --device cuda`
Expected: every progress line says `0 mensajes nuevos`, and `scoreboard.md` now has a `## Latencia` section.

- [ ] **Step 6: Save the preview.** Copy `bench_out/scoreboard.md` to `docs/benchmarks/<YYYY-MM-DD>-preliminar.md`, using the run date. Check that the PRELIMINAR banner is at the top.

- [ ] **Step 7: Append the run/read sections to `docs/benchmark.md`.**

````markdown

## Cómo correr

```bash
uv sync
uv run python scripts/bench.py sources                     # cuántos ejemplos hay por fuente y categoría
uv run python scripts/bench.py run                         # corre los modelos y escribe bench_out/scoreboard.md
uv run python scripts/bench.py run --include-unreviewed    # vista PRELIMINAR con filas sin revisar
uv run python scripts/bench.py latency --device cpu        # latencia en CPU (y --device cuda)
```

Todos los modelos son de código abierto (Apache-2.0) y corren en nuestras máquinas: ningún mensaje sale a una API externa. Un modelo nuevo entra al benchmark solo si cumple lo mismo; el test `test_every_contestant_is_open_source` lo controla.

Las predicciones quedan guardadas en `bench_out/predictions/`. Volver a correr solo evalúa lo nuevo: si revisaste o corregiste filas de la suite, solo se vuelven a evaluar esas. Con `--bootstrap 200` el cálculo de intervalos es más rápido.

## Cómo leer el scoreboard

- **AUROC [IC 95%]**: probabilidad de que el modelo puntúe más alto un mensaje dañino que uno sano. 0.5 es tirar una moneda y 1 es perfecto. El intervalo muestra cuánto puede variar por azar.
- **R · FPR**: cada modelo tiene un umbral, elegido para que marque por error al 5% de los mensajes sanos de la partición de calibración. R es qué parte de los dañinos detecta con ese umbral y FPR qué parte de los sanos marca en cada fuente. Si el FPR de una fuente se aleja mucho del 5%, ese umbral no se traslada bien a ese tipo de texto.
- **Diferencia contra el mejor**: si el intervalo incluye 0, no hay evidencia de que el mejor sea mejor de verdad.
- **pocos datos**: menos de 30 ejemplos de alguna clase; no sacar conclusiones.
- **suite-es/human vs. suite-es/generated**: si un modelo anda mucho mejor con las frases generadas, probablemente se parezca más al generador que al chat real.
- **Laya** se evalúa sin ajuste (zero-shot) y con preguntas sin pulir. La versión ajustada compite en el paso 2, junto con nuestros propios modelos.

## Fuentes y licencias

| Fuente | Idioma | Licencia |
|---|---|---|
| suite-es | es | propia |
| HateCheck multilingüe (es, pt) y HateCheck (en) | es/pt/en | CC BY 4.0 |
| ToLD-Br | pt | CC BY-SA 4.0 |
| OLID-BR | pt | CC BY 4.0 |

Quedaron afuera por licencia o acceso: OffendES (pide aceptar términos y su licencia se contradice), HatEval y HateBR (no comerciales) y el modelo de odio de pysentimiento (entrenado con HatEval).
````

- [ ] **Step 8: Add a README section.** In `README.md`, insert this section right before `## Project layout`:

````markdown
## Moderation benchmark

Before choosing a moderation model, `bench/` scores the candidates on one shared exam: Spanish, Portuguese and English messages labeled for insult, threat, identity hate, sexual harassment, profanity and target.
- **Sources:** HateCheck (es/pt/en), ToLD-Br and OLID-BR, plus a Rioplatense casino suite the team reviews (`suites/es_casino.csv`).
- **Contestants:** Laya zero-shot, the app's current Laya question, Detoxify multilingual and an mmBERT toxicity model. All are open-source and run locally; no paid or hosted APIs.

```bash
uv run python scripts/bench.py sources       # what the exam contains
uv run python scripts/bench.py run           # writes bench_out/scoreboard.md
uv run python scripts/bench.py latency --device cpu
```

Every model gets one operating threshold, set on a calibration split at 5% false positives. The report shows per-source AUROC with bootstrap confidence intervals, paired comparisons against the best model, and per-functionality accuracy. The labeling policy, the review guide and how to read the report are in [`docs/benchmark.md`](docs/benchmark.md) (in Spanish).
````

Also add these two lines to the `## Project layout` code block, after the `trainer/` entries:

```
bench/             moderation benchmark: sources, contestants, metrics, report
suites/            the hand-written Spanish suite reviewers edit
```

- [ ] **Step 9: Commit.** Summary: "add first preliminary scoreboard and benchmark docs".

```bash
git add docs/benchmarks/ docs/benchmark.md README.md
git commit
```

---

## Verification (end to end)

1. `uv run pytest`: the whole fast suite passes with no network access needed.
2. `uv run pytest -m slow`: real HateCheck, ToLD-Br and OLID-BR load with both `flag` classes; Laya, Detoxify and Horizon load and rank "sos un pelotudo de mierda" above a greeting.
3. `uv run python scripts/bench.py sources --include-unreviewed` shows the expected counts.
4. `uv run python scripts/bench.py run --include-unreviewed --device cuda` writes `bench_out/scoreboard.md` with the PRELIMINAR banner. A second run reports `0 mensajes nuevos` everywhere.
5. Edit one row's text in `suites/es_casino.csv` and set `reviewed=1`, then run `run` without `--include-unreviewed`. Only that row is predicted, and the banner disappears. Undo the edit afterwards.
6. `docs/benchmarks/<date>-preliminar.md` is committed. `git status` shows `bench_out/` untracked and ignored.

## After this plan (not part of it)

- **Spanish reviewers label the suite.** That means `reviewed=1`, at least 50 rows with `author=human`, and a second reviewer on ~150 rows checked with `bench.py agreement`. Then run without `--include-unreviewed`: **that is the first scoreboard that supports a decision.**
- **Step 2 plan:** fine-tune our own multi-label encoders (mmBERT-small, XLM-R-base, RoBERTuito for Spanish) and Laya. Deduplicate their training data against every benchmark source, including near-duplicates. Add the de-obfuscation normalizer and lexicon as an A/B on this benchmark.
- **Step 3 plan:** a cascade (normalizer → fast encoder → a self-hosted, open-weight LLM judge on the uncertain band), serving on the target hardware, and a feedback loop from reports and moderators. Judge candidates: Qwen3Guard and gpt-oss-safeguard, both Apache-2.0. Llama Guard's license is not OSI-approved, so it needs an explicit decision.
- **Data to look into:** the license of `piuba-bigdata/contextualized_hate_speech` (real Argentine data) as a second Spanish source, and OffendES if legal clears it.
