# Model Evaluation — Picking the Best Base Model for SFT + GRPO on AWS RL Env

## TL;DR

**Train `qwen2.5-coder-3b-instruct`.** It's the strongest candidate across every metric that matters for this task: highest exact-match rate, tightest outputs, and fast enough to not bottleneck GRPO rollouts. Full reasoning and per-model data below.

---

## 1. What this evaluation does

For each chat model loaded in LM Studio, we send 27 prompts drawn from our held-out validation split and measure how closely the model's output matches the canonical AWS CLI command that would solve the task. The goal is to pick the base model that:

1. **Starts strong** — already understands AWS CLI syntax, so SFT can focus on task correctness instead of format-locking
2. **Has headroom** — not so perfect that SFT overfits; not so weak that SFT can't help
3. **Is fast enough** — GRPO generates `G=8` rollouts per prompt × many prompts × many steps; inference cost compounds

This is a **format-and-correctness screen**. It does NOT measure:
- Whether the model can run a multi-step task against the live env (that's a separate integration test)
- Long-context behavior beyond ~500 tokens
- Post-SFT performance (only base-model zero-shot)

## 2. Eval methodology

### Prompts
- **Source**: `data/sft/aws_rl_sft.val.jsonl` (150 rows)
- **Coverage**: 3 examples per `(tier, source)` combo → **27 prompts per model**
- Combos cover: warmup+beginner+intermediate tiers × success_first_step + multi_step_continuation + failure_recovery + verification + hint_usage producers
- Each prompt is sent exactly as inference.py would send it: `system` + `user` messages from the dataset, no assistant turn

### Model invocation
- **Endpoint**: LM Studio at `http://localhost:1234/v1/chat/completions` (OpenAI-compatible)
- **temperature**: `0.0` (deterministic)
- **max_tokens**: `120` (enough for any valid AWS command; truncates runaway prose)
- **timeout**: `60s` per call

### Total budget
- 11 chat models × 27 prompts = **297 API calls**, completed in ~15 minutes

## 3. Metrics — what each column means

| Metric | What it measures | Why it matters |
|---|---|---|
| **`fmt%`** | Raw model output starts with `aws ` (no preamble, no fences, no prose) | Inference-time gate: [inference.py:93](../../inference.py#L93) rejects anything that doesn't start with `aws ` and replaces it with `aws help`. High `fmt%` = fewer wasted env steps. |
| **`+xtr%`** | After stripping markdown fences and leading prose, does the first `aws ...` line exist? | Measures "the model knows the answer but wraps it in junk." If `+xtr% >> fmt%`, the gap is all format noise — a simple regex in inference.py could recover most of it, OR SFT can lock it cheaply. |
| **`exact%`** | Extracted command matches the canonical command token-for-token | The hardest metric. Hits all the way down to exact flag values and escaping. This is the ceiling SFT has to reach. |
| **`svc%`** | Extracted command uses the same AWS service as canonical (e.g. both start with `aws s3api`) | Measures domain orientation: does the model know "this task calls for DynamoDB" even if it gets the exact operation wrong? |
| **`op%`** | Same AWS service AND same operation (e.g. both are `aws s3api create-bucket`) | Measures how close the model is to correct — it knows *what* to do, maybe not with *which* flags. This is the gap SFT closes most reliably. |
| **`lat`** | Mean seconds per call | Matters for GRPO rollout throughput. G=8 rollouts × 100 prompts × 5 steps = 4000 generations per training epoch. At 10s/call that's 11 hours; at 3s it's 3.3 hours. |
| **`len`** | Mean raw output length in characters | Proxy for verbosity. Lower = more concentrated signal for SFT loss; higher = model likes to explain itself (bad for this task). |

### Symbols in per-call logs
- **✓** — exact match with canonical command
- **~** — format valid (after extraction) but content doesn't match canonical
- **✗** — either no valid `aws ` line or the output is malformed

## 4. Full results — 11 models × 27 prompts each

```
Model                                  n errs  fmt%  +xtr%  exact%  svc%   op%   lat   len
--------------------------------------------------------------------------------------------
qwen2.5-coder-3b-instruct             27    0   85%   100%     41%   70%   63%  3.1s   86  ⭐
qwen/qwen3-4b-2507                    27    0  100%   100%     33%   74%   59% 10.4s  108
qwen2.5-coder-1.5b-instruct           27    0   81%    85%     22%   48%   44%  2.5s  110
smollm2-1.7b-instruct                 27    0   63%    63%      7%   63%   37%  2.1s   87
smollm-360m-instruct                  27    0    0%    63%      0%   26%    7%  1.7s  402
smollm2-135m-instruct                 27    0    0%    59%      0%   15%    7%  1.1s  337
smollm-360m-instruct-v0.2             27    0    0%    56%      0%   15%    7%  2.2s  364
smollm2-360m-instruct                 27    0   52%    52%      0%   48%   33%  1.0s  137
smollm-1.7b-instruct-v0.2             27    0    0%    37%      0%   15%   11%  3.9s  342
smollm2-360m (base)                   27    0    0%     0%      0%    0%    0%  1.7s  390
deepseek-r1-distill-qwen-1.5b         27    0    0%     0%      0%    0%    0%  4.1s    0†
```

*† DeepSeek-R1-Distill was truncated by `max_tokens=120` during its `<think>...</think>` reasoning phase. We re-ran it separately with `max_tokens=2048` — see section 6 for real numbers.*

## 5. Per-model verdicts

### ⭐ `qwen2.5-coder-3b-instruct` — **recommended**

**Evidence**
- **exact% = 41%** — highest of any model tested
- **op% = 63%** — best service+operation recognition; it knows *what* most tasks need
- **len = 86 chars** — tightest output in the test (even tighter than qwen3-4b at 108)
- **lat = 3.1s** — 3.4× faster than qwen3-4b with better accuracy
- Correctly handled `aws cognito-idp create-user-pool --pool-name app-users` (intermediate tier)
- Correctly handled `aws rds create-db-instance --db-instance-identifier app-database --engine mysql` (a notoriously long command)

**Weaknesses**
- `fmt% = 85%` (not 100%) — occasionally wraps commands in `'...'` quotes or adds a trailing period. SFT fixes this in one epoch.
- Sometimes picks the wrong operation within the right service (e.g. `create-user-pool-client` instead of `create-user-pool`). Failure-recovery rows in your SFT dataset address this directly.

**Training implications**
- Recommended LoRA config: **r=8, α=16, 2 epochs, lr=2e-4** — model is already strong enough that r=16 would memorize rather than generalize
- Expected post-SFT performance: exact% > 75%, op% > 90%
- Inference cost during GRPO: ~3× cheaper than qwen3-4b

---

### `qwen/qwen3-4b-2507` — strong runner-up

**Evidence**
- **fmt% = 100%** — the only model that never produces preamble, quotes, or fences
- **exact% = 33%**, **svc% = 74%** — still very good
- **lat = 10.4s** — 3× slower than qwen2.5-coder-3b due to 33% more parameters

**Weaknesses**
- The latency is a real problem for GRPO at scale — 10s × G=8 rollouts × 100 prompts = 2.2 hours per training step pair
- Lower `op%` than qwen2.5-coder-3b (59% vs 63%) despite being larger — suggests coder-tuning beats raw scale for this task

**Verdict**: use only if post-SFT evaluation on qwen2.5-coder-3b falls short of expectations. Otherwise the smaller coder model dominates.

---

### `qwen2.5-coder-1.5b-instruct` — the speed play

**Evidence**
- **fmt% = 81%**, **+xtr% = 85%**, **exact% = 22%**
- **lat = 2.5s** — fastest of the viable candidates
- 1.5B parameters — ~2× cheaper inference than the 3B

**Weaknesses**
- 22% exact-match is a real accuracy gap from the 3B (41%)
- Sometimes confuses related operations (e.g. `put-secret-value` instead of `create-secret`)

**Verdict**: keep as a fallback. If your GRPO budget is tight, the 2× throughput might justify the accuracy hit — but only after confirming SFT can close the gap. Recommended only if you plan to run many thousands of GRPO episodes.

---

### `smollm2-1.7b-instruct` — best of the SmolLMs, but not enough

**Evidence**
- **exact% = 7%** (2/27 correct) — only SmolLM variant above zero
- **svc% = 63%** — knows which service most tasks target
- Picks up service names fairly often but almost always with wrong operation or flags

**Weaknesses**
- A 34% accuracy gap to qwen2.5-coder-3b on the critical exact% metric
- Frequent hallucinations: `aws s3 mb s3://firehose-delivery/ --profile aws-dev-prod` (made-up profile flag)

**Verdict**: not worth training. The post-SFT ceiling will be limited by the base model's sparse AWS knowledge.

---

### `smollm2-135m-instruct` — surprising +xtr%, zero substance

**Evidence**
- **+xtr% = 59%** — emits `aws ` prefixed lines more often than half the larger SmolLMs
- **exact% = 0%**, **op% = 7%** — complete syntax salad behind the prefix

**Example outputs**
- `aws s3 ls --bucket=/path/to/s3 -o /path/to/s3-output.json -n notifications` (hallucinated flags for list-topics task)
- `aws elastic describe-cache-clusters --cluster=my_elastiCache` (wrong service name, fabricated flags)

**Verdict**: it produces convincing-looking CLI syntax but none of it is valid. A completely different failure mode from the 360M models (which dump prose) — and equally useless.

---

### `smollm-360m-instruct` / `smollm-360m-instruct-v0.2` / `smollm2-360m-instruct`

All three fail similarly:
- `fmt%` either 0% (dumps prose or Python code) or ~50% (emits quoted strings like `"'aws s3 ls'"`)
- `exact% = 0%` across the board
- Outputs often include markdown code fences, step-by-step narration, or hallucinated boto3 code

**Verdict**: ineligible. Format instability makes SFT expensive and the base capability is absent.

---

### `smollm-1.7b-instruct-v0.2` — size doesn't save it

**Evidence**
- Same parameter count as `smollm2-1.7b-instruct` but older / different training
- **+xtr% = 37%** vs. 63% for smollm2-1.7b — the training difference matters more than scale
- 0% exact match, 11% op match

**Verdict**: the newer smollm2-1.7b-instruct is strictly better; this variant has no role.

---

### `smollm2-360m` (base, not instruct)

**Evidence**
- 0% across every column
- Echoes the prompt back verbatim

**Verdict**: base models without instruction tuning are architecturally wrong for a chat-format SFT setup. Skip.

---

### `deepseek-r1-distill-qwen-1.5b` — wrong tool for this job

**Original run (max_tokens=120)**
- 0% across the board, 0-char outputs
- **Cause**: R1 models emit `<think>...</think>` reasoning blocks of 500-2000 tokens before their answer. 120 tokens truncated every response mid-thinking.

**Re-run (max_tokens=2048)**
- **exact% = 0/27** (still zero)
- **avg latency = 16.0s** (2-3× slower than qwen3-4b due to thinking overhead)
- 2 calls timed out at 60s
- Typical outputs: `aws s3 bucket-create --bucket data-pipeline` (invented op), `aws s3 topic --name Alerts` (wrong service), `aws iam checkRolePolicy` (hallucinated op name)

**Why it fails**
- R1-distill was trained on math and coding reasoning — not AWS CLI
- The `<think>` pattern doesn't summon domain knowledge that isn't in the base model
- Qwen-1.5B's AWS knowledge is sparse; wrapping it in reasoning tokens doesn't add substance

**Verdict**: only useful if you specifically want GRPO-with-thinking from day one AND are willing to do heavier SFT. For this task, qwen2.5-coder-3b + emergent reasoning during GRPO (R1-Zero style) is the cleaner path.

## 6. How to read the gap between `fmt%` and `+xtr%`

This gap tells you what kind of SFT each model needs:

- **`qwen/qwen3-4b-2507`**: `fmt% = +xtr% = 100%` → zero format-locking needed, SFT can focus entirely on task correctness
- **`qwen2.5-coder-3b`**: `85% → 100%` → small format tax (quotes, trailing punctuation); one epoch of SFT fixes it
- **`smollm-360m-instruct`**: `0% → 63%` → the model *knows* what to say but always wraps it in prose. A regex post-processor could salvage 63% without any training — but it's cheap signal to SFT on
- **`deepseek-r1-distill`**: `0% → 0%` → format-broken even with reasoning budget; not recoverable by regex

## 7. Overall ranking (for SFT + GRPO)

| Rank | Model | Train? | Reasoning |
|------|---|:---:|---|
| 1 | qwen2.5-coder-3b-instruct | ✅ | Best exact%, best op%, cleanest output, fast enough for GRPO |
| 2 | qwen/qwen3-4b-2507 | ⚠️ fallback | Perfect format but 3× slower and slightly worse content than #1 |
| 3 | qwen2.5-coder-1.5b-instruct | ⚠️ speed play | Strong for its size; train only if GRPO throughput is critical |
| 4 | smollm2-1.7b-instruct | ❌ | 34pt gap on exact% vs #1; ceiling too low |
| — | All smaller SmolLMs | ❌ | Format-broken, zero exact match, hallucinated syntax |
| — | smollm-1.7b-instruct-v0.2 | ❌ | Strictly dominated by smollm2-1.7b-instruct |
| — | deepseek-r1-distill-qwen-1.5b | ❌ | Wrong domain + latency 2× worse than #2 |

## 8. Caveats & limitations

- **27 prompts is a sample, not an exhaustive benchmark.** The error bars on exact% are ±5-10 percentage points. For close calls (like coder-3b vs qwen3-4b), rerun with `--max-per-combo 5` or higher before making the final call.
- **LM Studio latency is serving-architecture-dependent.** The 10s/call for qwen3-4b reflects Metal / llama.cpp on your local Mac. During actual training we'll run on CUDA via `transformers` (~100ms forward pass) or vLLM (~30ms), and the picture changes.
- **We only measure single-turn behavior.** Multi-step task completion (does the model actually solve the episode end-to-end?) requires running against the live env. This eval predicts first-step performance, which correlates well but isn't the same thing.
- **R1-distill was tested twice** — once with the default budget that truncated thinking, once with `max_tokens=2048`. The README table shows the truncated numbers; real performance is section 5's re-run.

## 9. Training implications — if you pick `qwen2.5-coder-3b-instruct`

- **LoRA**: `r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05` — lower rank than the default because the base model is already strong
- **Training**: `num_train_epochs=2, lr=2e-4, effective_batch=16, max_seq_length=512, lr_scheduler="cosine"` — shorter than the plan for Llama-3.1-8B; don't over-train
- **Expected post-SFT**: fmt% → 100%, op% → 90%+, exact% → 75%+
- **GRPO after SFT**: ~3× cheaper rollouts than qwen3-4b, so more exploration per compute budget

## 10. Files produced by this evaluation

- [model_eval_full.json](model_eval_full.json) — full per-call data (every prompt × every model × every response), 297 rows
- [model_eval_full.txt](model_eval_full.txt) — raw execution log (what was streamed to stdout during the run)
- [deepseek_r1_rerun.json](deepseek_r1_rerun.json) — R1-distill re-run data with `max_tokens=2048`
- [../eval_lm_studio_models.py](../eval_lm_studio_models.py) — the eval harness (reusable for post-SFT evaluation)

## 11. How to rerun this evaluation post-SFT

After training, save the merged model to LM Studio and rerun:

```bash
.venv/bin/python data/eval_lm_studio_models.py \
    --max-per-combo 5 \
    --out data/sft/model_eval_postsft.json
```

Compare the `exact%` and `op%` deltas vs the baseline in [model_eval_full.json](model_eval_full.json). A successful SFT run should see:
- `exact%`: 41% → 75%+
- `op%`: 63% → 90%+
- `fmt%`: 85% → 100%

If those deltas don't land, something's wrong with the training — not the dataset.
