"""Benchmark LM Studio models against a curated AWS-RL eval set.

Picks one example per (tier, source) combo from the val split, sends each
prompt to every chat model loaded in LM Studio, and reports which model is
the best candidate for SFT + GRPO.

Usage:
    .venv/bin/python data/eval_lm_studio_models.py \\
        --base-url http://localhost:1234/v1 \\
        --val data/sft/aws_rl_sft.val.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

EMBEDDING_HINT = ("embed", "embedding")


def load_eval_set(val_path: Path, max_per_combo: int = 1) -> list[dict]:
    """One row per (tier, source) combo from the val JSONL."""
    rows = [json.loads(l) for l in open(val_path)]
    seen: dict[tuple, int] = {}
    picks: list[dict] = []
    for r in rows:
        key = (r["difficulty"], r["source"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= max_per_combo:
            picks.append(r)
    return picks


def list_chat_models(client: OpenAI) -> list[str]:
    """Return chat-capable model ids (skip embeddings)."""
    out: list[str] = []
    for m in client.models.list().data:
        mid = m.id.lower()
        if any(h in mid for h in EMBEDDING_HINT):
            continue
        out.append(m.id)
    return out


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 120,
    timeout: float = 60.0,
) -> tuple[str, float, str | None]:
    """Return (completion_text, latency_s, error_or_None)."""
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, time.time() - t0, None
    except Exception as exc:
        return "", time.time() - t0, f"{type(exc).__name__}: {exc}"


def extract_command(raw: str) -> str:
    """Strip markdown fences, code blocks, leading/trailing prose to get the command."""
    text = raw.strip()
    # Strip ```...``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()
    # Take first line that starts with 'aws '
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("aws "):
            return line
    return text  # no aws line found — return as-is for diagnosis


def score(
    completion: str,
    expected: str,
) -> dict[str, Any]:
    extracted = extract_command(completion)
    raw_stripped = completion.strip()
    return {
        "format_ok": raw_stripped.startswith("aws "),
        "format_ok_after_extract": extracted.startswith("aws "),
        "exact_match": extracted == expected.strip(),
        "service_match": (
            extracted.split()[1:2] == expected.split()[1:2]
            if len(extracted.split()) >= 2 and len(expected.split()) >= 2
            else False
        ),
        "operation_match": (
            extracted.split()[2:3] == expected.split()[2:3]
            if len(extracted.split()) >= 3 and len(expected.split()) >= 3
            else False
        ),
        "raw_len_chars": len(completion),
        "extracted": extracted[:120],
    }


def run_benchmark(
    client: OpenAI,
    models: list[str],
    eval_set: list[dict],
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {m: [] for m in models}
    for i, task in enumerate(eval_set):
        expected = task["messages"][2]["content"]
        messages_in = task["messages"][:2]  # system + user, no assistant
        tier = task["difficulty"]
        source = task["source"]
        print(f"\n[{i+1}/{len(eval_set)}] tier={tier} source={source} task_id={task['task_id']}")
        print(f"  expected: {expected[:90]!r}")
        for model in models:
            completion, latency, err = call_model(client, model, messages_in)
            if err:
                row = {
                    "tier": tier, "source": source, "task_id": task["task_id"],
                    "completion": "", "error": err, "latency_s": round(latency, 2),
                    "format_ok": False, "format_ok_after_extract": False,
                    "exact_match": False, "service_match": False,
                    "operation_match": False, "raw_len_chars": 0, "extracted": "",
                }
            else:
                s = score(completion, expected)
                row = {
                    "tier": tier, "source": source, "task_id": task["task_id"],
                    "completion": completion, "error": None,
                    "latency_s": round(latency, 2),
                    **s,
                }
            results[model].append(row)
            flag = "✓" if row.get("exact_match") else ("~" if row.get("format_ok_after_extract") else "✗")
            print(f"    {flag} {model:<35} {latency:5.1f}s  {row.get('extracted','')[:70]!r}")
    return results


def aggregate(results: dict[str, list[dict]]) -> list[dict]:
    agg = []
    for model, rows in results.items():
        n = len(rows)
        if n == 0:
            continue
        agg.append({
            "model": model,
            "n": n,
            "errors": sum(1 for r in rows if r.get("error")),
            "format_ok_pct": round(sum(1 for r in rows if r["format_ok"]) / n, 2),
            "format_after_extract_pct": round(
                sum(1 for r in rows if r["format_ok_after_extract"]) / n, 2
            ),
            "exact_match_pct": round(sum(1 for r in rows if r["exact_match"]) / n, 2),
            "service_match_pct": round(sum(1 for r in rows if r["service_match"]) / n, 2),
            "operation_match_pct": round(sum(1 for r in rows if r["operation_match"]) / n, 2),
            "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
            "avg_len_chars": round(sum(r["raw_len_chars"] for r in rows) / n, 1),
        })
    agg.sort(
        key=lambda d: (d["format_after_extract_pct"], d["exact_match_pct"], -d["avg_latency_s"]),
        reverse=True,
    )
    return agg


def print_table(agg: list[dict]) -> None:
    print("\n" + "=" * 110)
    print(f"{'Model':<36} {'n':>3} {'errs':>4} {'fmt%':>5} {'+xtr%':>6} {'exact%':>7} {'svc%':>5} {'op%':>5} {'lat':>5} {'len':>5}")
    print("-" * 110)
    for r in agg:
        print(
            f"{r['model']:<36} {r['n']:>3} {r['errors']:>4} "
            f"{int(r['format_ok_pct']*100):>4}% {int(r['format_after_extract_pct']*100):>5}% "
            f"{int(r['exact_match_pct']*100):>6}% {int(r['service_match_pct']*100):>4}% "
            f"{int(r['operation_match_pct']*100):>4}% {r['avg_latency_s']:>4.1f}s {int(r['avg_len_chars']):>4}"
        )
    print("=" * 110)
    print("Column legend:")
    print("  fmt%    — raw output starts with 'aws ' (no preamble, no fences)")
    print("  +xtr%   — starts with 'aws ' after stripping fences/prose")
    print("  exact%  — extracted command matches canonical exactly")
    print("  svc%    — same AWS service (e.g. s3, dynamodb)")
    print("  op%     — same operation (e.g. create-bucket)")
    print("  lat     — mean seconds per call  |  len — mean raw chars")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--val", type=Path, default=Path("data/sft/aws_rl_sft.val.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/sft/model_eval_results.json"))
    ap.add_argument("--max-per-combo", type=int, default=1)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="lm-studio")
    models = list_chat_models(client)
    print(f"Found {len(models)} chat models: {models}")

    eval_set = load_eval_set(args.val, args.max_per_combo)
    print(f"Eval set: {len(eval_set)} prompts (one per (tier, source) combo)")

    results = run_benchmark(client, models, eval_set)
    agg = aggregate(results)
    print_table(agg)

    with open(args.out, "w") as f:
        json.dump({"aggregate": agg, "per_call": results}, f, indent=2)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
