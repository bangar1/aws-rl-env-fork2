"""Push the generated SFT JSONL splits to HuggingFace Hub as a proper dataset.

After upload, consumers can load it with:

    from datasets import load_dataset
    ds = load_dataset("<your-user>/aws-rl-sft")
    ds["train"]        # 1500 rows
    ds["validation"]   # 150 rows
    ds["reserve"]      # 200 held-out rows

Prerequisites:
    pip install datasets>=2.19 huggingface_hub
    export HF_TOKEN=hf_...           # or `huggingface-cli login`

Usage:
    python data/upload_sft_to_hf.py --repo-id <user>/aws-rl-sft
    python data/upload_sft_to_hf.py --repo-id <user>/aws-rl-sft --private
    python data/upload_sft_to_hf.py --repo-id <user>/aws-rl-sft --skip-push    # dry run
    python data/upload_sft_to_hf.py --repo-id Sizzing/aws-rl-sft --private --token hf_**** # upload to an org repo with explicit token
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` looking for server/services/tasks/ as a sentinel."""
    for p in [start, *start.parents]:
        if (p / "server" / "services" / "tasks").is_dir():
            return p
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
SFT_DIR = REPO_ROOT / "data" / "sft"

SPLIT_FILES: dict[str, str] = {
    "train": "aws_rl_sft.train.jsonl",
    "validation": "aws_rl_sft.val.jsonl",
    "reserve": "aws_rl_sft.reserve.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dataset_dict(sft_dir: Path):
    """Build a DatasetDict from the JSONL splits."""
    from datasets import Dataset, DatasetDict

    splits = {}
    for split, fname in SPLIT_FILES.items():
        path = sft_dir / fname
        if not path.exists():
            print(f"  skip split '{split}' — {path} not found")
            continue
        rows = load_jsonl(path)
        ds = Dataset.from_list(rows)
        splits[split] = ds
        print(f"  loaded '{split}': {len(rows)} rows, columns={list(ds.column_names)}")
    return DatasetDict(splits)


DATASET_CARD = """---
language:
- en
license: apache-2.0
size_categories:
- 1K<n<10K
task_categories:
- text-generation
tags:
- aws
- aws-cli
- sft
- lora
- agentic
- rl
- grpo
- tool-use
pretty_name: AWS RL Env SFT
---

# AWS RL Env — SFT Dataset

Supervised fine-tuning dataset for training an LLM agent that operates AWS
infrastructure via the CLI. Built for the **aws-rl-env** reinforcement-learning
environment, which emulates 34 AWS services in-container (MiniStack) and rewards
agents for completing cloud-operations tasks via single-command steps.

Designed as the **cold-start phase** of an SFT → GRPO pipeline:
1. **SFT with LoRA** (this dataset) — command-only assistant targets, lock output format
2. **GRPO on curriculum** — refine policy with online env reward, optionally emerge `<think>` reasoning

## Schema

Each row is one `(state → command)` decision, formatted as HuggingFace chat messages.
Directly compatible with `trl.SFTTrainer` (auto-detects `messages` column and
applies the tokenizer's chat template).

```python
{
    "task_id": int,
    "difficulty": "warmup" | "beginner" | "intermediate" | "advanced" | "expert",
    "source": "success_first_step" | "multi_step_continuation" | "failure_recovery" | "verification" | "hint_usage",
    "step_idx": int,
    "messages": [
        {"role": "system",    "content": "<system prompt>"},
        {"role": "user",      "content": "TASK: ... Step: N ..."},
        {"role": "assistant", "content": "aws ..."},
    ],
}
```

## Composition (by source)

| Source | Share | What it teaches |
|---|---:|---|
| `success_first_step` | ~55% | Canonical command at step 0 given empty state |
| `multi_step_continuation` | ~20% | Step N>0 with prior command history |
| `failure_recovery` | ~15% | Correct command after a plausible mistake (wrong-op, missing-arg, s3-vs-s3api, typo, etc.) |
| `verification` | ~5% | Read-only verify command after task completion |
| `hint_usage` | ~5% | Edge case: assistant requests hint via `aws help --task-hint` |

## Composition (by tier)

| Tier | Share |
|---|---:|
| warmup | ~30% |
| beginner | ~25% |
| intermediate | ~44% |
| advanced | 0% — deferred to GRPO (dynamic resource IDs can't be safely synthesized offline) |
| expert | 0% — deferred to GRPO (policy-crafting / security audits benefit from env reward) |

## Splits

| Split | Rows | Purpose |
|---|---:|---|
| `train` | 1500 | LoRA SFT training |
| `validation` | 150 | Eval loss, early stopping |
| `reserve` | 200 | Held-out; use only if train proves insufficient |

## Quickstart

### Load

```python
from datasets import load_dataset

ds = load_dataset("<your-user>/aws-rl-sft")
print(ds)
```

### Filter by source or tier

```python
easy = ds["train"].filter(lambda r: r["difficulty"] in ("warmup", "beginner"))
recovery = ds["train"].filter(lambda r: r["source"] == "failure_recovery")
```

### Train with `trl.SFTTrainer` + LoRA

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")

ds = load_dataset("<your-user>/aws-rl-sft")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    ),
    args=SFTConfig(
        output_dir="./sft-ckpt",
        max_seq_length=2048,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=10,
        bf16=True,
        packing=False,
    ),
)
trainer.train()
trainer.save_model("./sft-ckpt/final")
```

## Generation notes

- **Fully synthetic, no teacher LLM required.** Canonical commands were pulled from
  the env's own test suite (`tests_tasks/test_*.py`), where each task's command
  sequence is already verified to pass the grader with reward 1.0.
- **Failure-recovery rows** use a 5-mistake catalog (wrong-op, missing-arg,
  wrong-service, s3-vs-s3api confusion, character-swap typo) paired with realistic
  AWS CLI error messages.
- **Prompt variance** injected via reward jitter (±0.1), history-window trimming,
  and sampled reset-state outputs so dedup-on-exact-prompt still produces enough
  unique rows.

## License

Apache 2.0. AWS CLI commands themselves are public interface; assistant targets
were generated deterministically from the env's grader test suite.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", required=True, help="HF repo id, e.g. username/aws-rl-sft")
    ap.add_argument("--private", action="store_true", help="Create as private repo")
    ap.add_argument("--sft-dir", type=Path, default=SFT_DIR)
    ap.add_argument("--token", default=None, help="HF token (falls back to HF_TOKEN env var)")
    ap.add_argument(
        "--skip-push",
        action="store_true",
        help="Build + save locally, don't upload (useful for testing)",
    )
    args = ap.parse_args()

    try:
        from datasets import DatasetDict  # noqa: F401
    except ImportError:
        raise SystemExit(
            "The 'datasets' library is required. Install it with:\n"
            "  pip install datasets>=2.19"
        )

    token = args.token or os.getenv("HF_TOKEN")
    if not token and not args.skip_push:
        raise SystemExit(
            "No HF token found. Either:\n"
            "  export HF_TOKEN=hf_...\n"
            "  # or\n"
            "  huggingface-cli login\n"
            "  # or pass --token explicitly\n"
            "  # or use --skip-push to build locally without uploading"
        )

    print(f"Loading JSONL splits from {args.sft_dir}...")
    ds_dict = build_dataset_dict(args.sft_dir)
    if not ds_dict:
        raise SystemExit(
            f"No splits loaded from {args.sft_dir}. "
            "Run build_sft_dataset.py first."
        )

    if args.skip_push:
        local_path = args.sft_dir / "hf_dataset_preview"
        ds_dict.save_to_disk(str(local_path))
        print(f"\n--skip-push: saved DatasetDict to {local_path}")
        print("Inspect with: datasets.load_from_disk('{0}')".format(local_path))
        print("\nOne sample row from 'train':")
        print(json.dumps(ds_dict["train"][0], indent=2)[:800])
        return

    from huggingface_hub import HfApi, login

    login(token=token)
    print(f"\nPushing to https://huggingface.co/datasets/{args.repo_id}")
    print(f"  private={args.private}")
    ds_dict.push_to_hub(args.repo_id, private=args.private, token=token)

    api = HfApi(token=token)
    readme_bytes = DATASET_CARD.encode("utf-8")
    api.upload_file(
        path_or_fileobj=readme_bytes,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print(f"\nDone. https://huggingface.co/datasets/{args.repo_id}")
    print("\nConsumer usage:")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('{args.repo_id}')")


if __name__ == "__main__":
    main()
