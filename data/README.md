# `data/` — SFT Dataset Generation & Base-Model Selection

[← back to main README](../README.md)

This directory holds the SFT training corpus, the dataset generator that produced it, and the rigorous benchmark we used to pick the base model. Together they answer two questions a hackathon judge should be able to verify in under five minutes:

1. **What did we train on?** A 1,500-row synthetic SFT corpus with five trajectory types covering success, continuation, failure recovery, verification, and hint usage. ([§1](#1-sft-dataset-generation))
2. **Why this base model?** A reproducible 11-model benchmark across 27 held-out prompts. **Qwen2.5-Coder-3B-Instruct** wins on every metric that matters. ([§5](#5-base-model-selection-overview))

> ![Top 4 candidate models on the held-out benchmark](../docs/figures/model_eval_chart.png)

---

## Table of contents

1. [SFT dataset generation](#1-sft-dataset-generation)
2. [Five trajectory types](#2-five-trajectory-types)
3. [Tier weighting](#3-tier-weighting)
4. [Dataset format & artifacts](#4-dataset-format--artifacts)
5. [Base-model selection — overview](#5-base-model-selection-overview)
6. [Eval harness](#6-eval-harness)
7. [HuggingFace publishing](#7-huggingface-publishing)
8. [Files in this directory](#8-files-in-this-directory)

---

## 1. SFT dataset generation

[data/build_sft_dataset.py](build_sft_dataset.py) — 27 KB, single-script generator.

### Approach

The dataset is **synthetically generated** but grounded in canonical solutions extracted from our integration test suite. Two design decisions worth flagging to judges:

#### AST-based extraction, not pytest execution

Each `tests_tasks/test_<tier>_tasks.py` file has a top-level constant (`WARMUP_COMMANDS`, `BEGINNER_COMMANDS`, …) mapping `task_id → canonical AWS CLI command`. We extract these via Python's `ast` module — we do **not** execute the test file. Reasons:

1. `pytest` fixtures would spin up a MiniStack, hit AWS APIs, and add 30+ seconds of overhead per generation run.
2. Static extraction is deterministic — no flake risk. The dataset is reproducible bit-for-bit given a seed.
3. The canonical solutions are intentionally simple constant declarations that AST can parse without import side effects.

#### Plausible-output simulation

When generating multi-step continuations, we don't have a real MiniStack response to feed back into the user message — we have to fabricate one. The generator maps each AWS operation (`list-buckets`, `create-table`, `describe-instances`, …) to a JSON template, then interpolates the right resource names from the task. So an `aws s3api list-buckets` step in the user prompt history has output like:

```json
{"Buckets":[{"Name":"my-app-data","CreationDate":"2026-04-15T..."}]}
```

…instead of the empty `{"Buckets":[]}` you'd get from a fresh MiniStack. This is the difference between the SFT model learning "first step, always answer with the canonical command" (degenerate) and "first step depends on what's already been done" (correct).

### Dynamic-ID filtering

Some tests reference resources whose IDs only exist at runtime — security groups (`sg-…`), subnets (`subnet-…`), VPCs (`vpc-…`), instance IDs (`i-…`). These commands cannot be deterministically captured by static extraction. The generator skips any task whose canonical command contains those patterns. The result: 72 unique tasks make it into the train split (out of 134 total tasks), all of which are deterministically reproducible.

---

## 2. Five trajectory types

The SFT corpus mixes five distinct trajectory shapes so the model learns to handle real multi-turn agent behavior, not just one-shot question answering. Actual proportions (from [data/sft/dataset_stats.json](sft/dataset_stats.json)):

| Source                     | Train pct (target) | Train rows | What the model sees                                                                       |
|----------------------------|:------------------:|:----------:|-------------------------------------------------------------------------------------------|
| `success_first_step`       | 55.1% (55%)        | 826        | User → Task description → assistant emits the canonical command                           |
| `multi_step_continuation`  | 20.1% (20%)        | 301        | User → Task description + a baked-in history of N-1 prior commands and their outputs → assistant emits step N |
| `failure_recovery`         | 15.5% (15%)        | 232        | User → Task description + step 1 of a wrong command and its simulated error → assistant emits the recovery command |
| `verification`             | 4.5% (5%)          | 67         | User → Task already complete → assistant emits a read-only verification command           |
| `hint_usage`               | 4.9% (5%)          | 74         | User → Task description → assistant emits `aws help --task-hint` (the agent action that requests a hint) |

Why include the last four sources at all?

- **`multi_step_continuation`** trains continuation behavior. Without it, the model overfits to step 1 and degrades on later turns.
- **`failure_recovery`** teaches the model that a typo / wrong command is recoverable. The reward signal during GRPO is dense — the model needs to know what "try again" looks like.
- **`verification`** trains the model to recognize when a task is done and respond appropriately. Production agents must distinguish "do something" from "confirm it's done".
- **`hint_usage`** lets the model learn that `aws help --task-hint` is the in-environment way to request help, not just a literal CLI command.

---

## 3. Tier weighting

[data/build_sft_dataset.py:54-60](build_sft_dataset.py) — sampling weights:

| Tier         | Weight | Train rows | Why                                                                                |
|--------------|:------:|:----------:|------------------------------------------------------------------------------------|
| warmup       | 0.50   | 456        | Most rows. Format-locks the model on the simplest possible "aws X list" pattern.   |
| beginner     | 0.30   | 378        | Single-resource creation — bread and butter.                                       |
| intermediate | 0.15   | 666 *      | Multi-step workflows. Note actual count > target because each task contributes more rows via multi_step_continuation. |
| advanced     | 0.05   | 0          | Cross-service architectures. Filtered out post-extraction (most have dynamic IDs). |
| expert       | 0.00   | 0          | SRE / drift / security-posture. **Intentionally excluded from SFT.**               |

> **Why expert tier is excluded from SFT.** The expert tasks (drift detection, security audits) have *randomized* state checks — there is no canonical command sequence. Trying to SFT on them would teach the model a particular fix script that is *wrong* on most episodes. These tasks are reserved for GRPO, where the env's `state_checks` reward signal handles the randomization correctly.

`*` Intermediate row count exceeds the simple weight because the multi-step trajectory generator naturally produces multiple rows per task (one for step 1, step 2, etc.).

---

## 4. Dataset format & artifacts

### JSONL chat-message schema

```json
{
  "messages": [
    {"role": "system", "content": "You are an AWS cloud engineer interacting with a real AWS environment via CLI..."},
    {"role": "user", "content": "TASK: Create an S3 bucket named my-app-data and enable versioning on it.\n\nPREVIOUS COMMANDS:\n[1] $ aws s3 mb s3://my-app-data\n    output: make_bucket: my-app-data\n    reward: 0.50\n\n---\n\nCURRENT OBSERVATION:\nProgress: 0.50  Achieved: False  Step: 2"},
    {"role": "assistant", "content": "aws s3api put-bucket-versioning --bucket my-app-data --versioning-configuration Status=Enabled"}
  ],
  "difficulty": "intermediate",
  "source": "multi_step_continuation",
  "task_id": 42
}
```

Every row carries the `difficulty`, `source`, and `task_id` metadata — useful for filtering, ablations, and debugging.

### Artifacts

[data/sft/](sft/):

| File                                                         | Size  | Rows  | Unique tasks | Use                                            |
|--------------------------------------------------------------|------:|------:|:------------:|------------------------------------------------|
| [aws_rl_sft.train.jsonl](sft/aws_rl_sft.train.jsonl)         | 2.2 MB | 1,500 | 72           | SFT training                                   |
| [aws_rl_sft.val.jsonl](sft/aws_rl_sft.val.jsonl)             | 218 KB | 150   | 63           | SFT validation; basis for [MODEL_EVALUATION.md](sft/MODEL_EVALUATION.md) |
| [aws_rl_sft.reserve.jsonl](sft/aws_rl_sft.reserve.jsonl)     | 294 KB | 200   | 66           | Held-out reserve for post-SFT regression checks |
| [dataset_stats.json](sft/dataset_stats.json)                 | 3.4 KB | —     | —            | Per-split source/tier/task breakdowns          |
| [MODEL_EVALUATION.md](sft/MODEL_EVALUATION.md)               | 15 KB  | —     | —            | Full model-selection writeup ([§5](#5-base-model-selection-overview)) |
| [model_eval_full.json](sft/model_eval_full.json)             | 209 KB | 297   | —            | Per-call eval data (11 models × 27 prompts)    |
| [deepseek_r1_rerun.json](sft/deepseek_r1_rerun.json)         | 5.3 KB | 27    | —            | DeepSeek R1 re-run with `max_tokens=2048`      |

---

## 5. Base-model selection — overview

This is the most rigorous decision in the whole project. Full reasoning, per-model verdicts, and methodology lives in **[data/sft/MODEL_EVALUATION.md](sft/MODEL_EVALUATION.md)** — a 270-line standalone report. Read it before judging the project's technical depth; it's what convinces us we're training the right thing.

The 30-second summary:

| Model                          | exact% | op%  | fmt%   | Latency | Verdict                              |
|--------------------------------|:-----:|:----:|:------:|:-------:|--------------------------------------|
| **qwen2.5-coder-3b-instruct**  | **41%** | **63%** | 85% | **3.1s**  | ✅ Train this. Highest exact, fastest viable. |
| qwen/qwen3-4b-2507             | 33%   | 59%  | 100%   | 10.4s   | Fallback. Perfect format, 3× slower.  |
| qwen2.5-coder-1.5b-instruct    | 22%   | 44%  | 81%    | 2.5s    | Speed play if GRPO budget tight.      |
| smollm2-1.7b-instruct          | 7%    | 37%  | 63%    | 2.1s    | ❌ Ceiling too low.                   |
| (7 more)                       | 0%    | …    | …      | …       | ❌ Format-broken or wrong domain.      |

> ![Per-model comparison: 5 quality metrics + latency](../docs/figures/model_eval_chart.png)

What the metrics mean:

- **`fmt%`**: raw output starts with `aws ` (no preamble, fences, or quotes). The agent's [inference.py:93](../inference.py) gate rejects everything else.
- **`+xtr%`**: `fmt%` after stripping markdown fences. Gap to `fmt%` = "model knows the answer, wrapping it in junk".
- **`exact%`**: extracted command matches canonical token-for-token. The hardest metric.
- **`svc%`**: same AWS service as canonical. Domain orientation.
- **`op%`**: same service AND operation. The gap SFT closes most reliably.

The full table (11 models, 9 metrics, per-call logs) is in [data/sft/model_eval_full.json](sft/model_eval_full.json) — 297 records.

---

## 6. Eval harness

[data/eval_lm_studio_models.py](eval_lm_studio_models.py) — 9.9 KB, reusable.

- Calls each chat model loaded in LM Studio at `http://localhost:1234/v1/chat/completions` (OpenAI-compatible API)
- Sends the same 27 held-out prompts to each model
- Extracts `aws ...` from the response (stripping fences / preamble)
- Compares against the canonical command from the val split
- Writes per-call detail + aggregate metrics to JSON

To re-run post-SFT:

```bash
.venv/bin/python data/eval_lm_studio_models.py \
    --max-per-combo 5 \
    --out data/sft/model_eval_postsft.json
```

A successful SFT run should see (predictions from [MODEL_EVALUATION.md §11](sft/MODEL_EVALUATION.md), and **actuals from our reference SFT run**):

| Metric    | Base  | Target  | **Actual (post-SFT)** |
|-----------|:-----:|:-------:|:---------------------:|
| `exact%`  | 39%   | 75%+    | **88.9%** ✅          |
| `op%`     | 61%   | 90%+    | **88.9%** ≈           |
| `svc%`    | 78%   | —       | **88.9%**             |
| `fmt%`    | 33%   | 100%    | **100.0%** ✅         |
| latency   | 2.03s | —       | **1.40s** (faster)    |

Every target from MODEL_EVALUATION.md is hit or essentially hit. Format compliance is now perfect; exact-match jumped 50 pp; the model is faster *and* tighter.

> ![Base vs SFT comparison (eval metrics)](../docs/figures/base_vs_sft_success.png)
> ![Single-step eval base vs SFT](../docs/figures/single_step_eval.png)

---

## 7. HuggingFace publishing

[data/upload_sft_to_hf.py](upload_sft_to_hf.py) — pushes the JSONL splits to HuggingFace Hub:

| Split    | Hub repo                                            |
|----------|-----------------------------------------------------|
| train    | `Sizzing/aws-rl-sft-qwen25coder3b-train`            |
| val      | `Sizzing/aws-rl-sft-qwen25coder3b-val`              |
| reserve  | `Sizzing/aws-rl-sft-qwen25coder3b-reserve`          |

The trained SFT adapter (output of [train/train_sft_lora.ipynb](../train/train_sft_lora.ipynb)) is published separately at:

- `Sizzing/aws-rl-sft-qwen25coder3b-adapter`

GRPO training picks it up by setting `SFT_ADAPTER = "Sizzing/aws-rl-sft-qwen25coder3b-adapter"` in [aws_rl_env_colab.ipynb](../aws_rl_env_colab.ipynb).

---

## 8. Files in this directory

| File                                                               | Purpose                                                            |
|--------------------------------------------------------------------|--------------------------------------------------------------------|
| [build_sft_dataset.py](build_sft_dataset.py)                       | Generator — AST extraction + 5 trajectory types + plausible outputs |
| [eval_lm_studio_models.py](eval_lm_studio_models.py)               | Base-model benchmark harness (LM Studio API)                       |
| [upload_sft_to_hf.py](upload_sft_to_hf.py)                         | Push the SFT splits to HuggingFace                                 |
| [sft/aws_rl_sft.train.jsonl](sft/aws_rl_sft.train.jsonl)           | 1,500 SFT training rows                                            |
| [sft/aws_rl_sft.val.jsonl](sft/aws_rl_sft.val.jsonl)               | 150 validation rows                                                |
| [sft/aws_rl_sft.reserve.jsonl](sft/aws_rl_sft.reserve.jsonl)       | 200 reserve rows                                                   |
| [sft/dataset_stats.json](sft/dataset_stats.json)                   | Per-split source / tier / task counts                              |
| [sft/MODEL_EVALUATION.md](sft/MODEL_EVALUATION.md)                 | **The base-model selection report (read this)**                    |
| [sft/model_eval_full.json](sft/model_eval_full.json)               | Per-call eval data (11 models × 27 prompts)                        |
| [sft/deepseek_r1_rerun.json](sft/deepseek_r1_rerun.json)           | R1 re-run with extended `max_tokens`                               |

---

## See also

- [Main README](../README.md)
- [data/sft/MODEL_EVALUATION.md](sft/MODEL_EVALUATION.md) — full base-model selection writeup
- [train/README.md](../train/README.md) — how this dataset is consumed by SFT training
- [compare/README.md](../compare/README.md) — how the trained model is benchmarked vs the base
- [server/services/tasks/](../server/services/tasks/) — source of truth for task definitions (the YAML the generator reads)
- [tests_tasks/](../tests_tasks/) — canonical solutions the generator extracts via AST
