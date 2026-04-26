"""Generate the 4 new PNG figures embedded in blog.md.

Outputs (idempotent):
    docs/figures/blog_hero.png
    docs/figures/tier_pyramid.png
    docs/figures/dataset_composition.png
    docs/figures/reward_components.png

Run from repo root:
    .venv/bin/python scripts/generate_blog_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PINK = "#ff4f8b"
PINK_DARK = "#c81b5a"
INK = "#1a1a1a"
SLATE = "#525a66"
PAPER = "#fff7fa"
GRID = "#e8d6df"

PALETTE = ["#3a86ff", "#8338ec", "#ff006e", "#fb5607", "#ffbe0b"]


def _save(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def hero() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(0.45, 4.55, "AWS Cloud Operations RL", fontsize=15,
            color=PINK_DARK, fontweight="bold", family="DejaVu Sans")
    ax.text(0.45, 3.85, "From Cloud Chaos to Capable Agents",
            fontsize=30, color=INK, fontweight="bold", family="DejaVu Sans")
    ax.text(0.45, 3.25, "Training an LLM SRE on 120+ AWS Tasks with SFT \u2192 GRPO",
            fontsize=15, color=SLATE, family="DejaVu Sans", style="italic")

    stats = [
        ("120+", "AWS tasks\n5 tiers + drift"),
        ("8\u00d7", "parallel rollouts\n1 GPU"),
        ("8", "anti-hacking\nlayers"),
        ("39\u219289%", "exact-match\npost-SFT"),
    ]
    box_w = 2.55
    gap = 0.2
    start_x = 0.45
    y = 0.55
    h = 2.1
    for i, (big, small) in enumerate(stats):
        x = start_x + i * (box_w + gap)
        box = FancyBboxPatch(
            (x, y), box_w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.5, edgecolor=PINK, facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + h * 0.62, big,
                fontsize=26, color=PINK_DARK, fontweight="bold",
                ha="center", va="center")
        ax.text(x + box_w / 2, y + h * 0.22, small,
                fontsize=10.5, color=SLATE, ha="center", va="center")

    ax.text(11.55, 4.55, "OpenEnv Hackathon  \u2022  Apr 2026",
            fontsize=10, color=SLATE, ha="right", style="italic")

    _save(fig, "blog_hero.png")


def tier_pyramid() -> None:
    # Top of pyramid (apex, narrow, hardest) \u2192 bottom (base, widest, easiest).
    tiers_top_down = [
        ("Expert",       24, "30%", "state_checks",         PALETTE[2]),
        ("Advanced",     25, "30%", "multi_step+services",  PALETTE[1]),
        ("Intermediate", 25, "20%", "multi_step",           PALETTE[0]),
        ("Beginner",     25, "10%", "resource_creation",    "#06b6d4"),
        ("Warmup",       25, "10%", "command_match",        "#22c55e"),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                  gridspec_kw={"width_ratios": [3.2, 1]})
    fig.patch.set_facecolor("white")

    n = len(tiers_top_down)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.2, n + 0.4)
    ax.axis("off")
    ax.set_title("Curriculum: 124 tasks across 5 tiers", fontsize=15,
                 fontweight="bold", color=INK, pad=12)

    for i, (name, count, chaos, strat, color) in enumerate(tiers_top_down):
        # i=0 \u2192 apex (top, narrowest); i=n-1 \u2192 base (bottom, widest)
        y_top = n - i
        y_bot = n - i - 1
        half_top = 0.45 + 0.55 * (i / (n - 1))           # narrow at apex
        half_bot = 0.45 + 0.55 * ((i + 1) / (n - 1))     # wider at base
        ax.add_patch(
            mpatches.Polygon(
                [(-half_bot, y_bot), (half_bot, y_bot),
                 (half_top, y_top), (-half_top, y_top)],
                closed=True, facecolor=color, edgecolor="white",
                linewidth=2, alpha=0.95,
            )
        )
        y_mid = (y_top + y_bot) / 2
        ax.text(0, y_mid + 0.18, name, fontsize=14, fontweight="bold",
                color="white", ha="center", va="center")
        ax.text(0, y_mid - 0.18,
                f"{count} tasks  \u00b7  chaos {chaos}  \u00b7  {strat}",
                fontsize=9.5, color="white", ha="center", va="center", alpha=0.97)

    # Drift sidebar (right panel)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, n + 0.4)
    ax2.axis("off")
    ax2.set_title("Adversarial track", fontsize=13, fontweight="bold",
                  color=INK, pad=12)

    box = FancyBboxPatch(
        (0.08, 1.7), 0.84, 1.7,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=PINK, edgecolor=PINK_DARK, linewidth=2, alpha=0.92,
    )
    ax2.add_patch(box)
    ax2.text(0.5, 3.0, "Drift", fontsize=20, fontweight="bold",
             color="white", ha="center")
    ax2.text(0.5, 2.6, "9 tasks", fontsize=12, color="white", ha="center")
    ax2.text(0.5, 2.05, "2\u20133 mutations\nrandomized\nper episode",
             fontsize=9.5, color="white", ha="center", va="center")

    ax2.text(0.5, 0.85,
             "Promotion paths\n\u2014\nstandard: min episodes + rate\nfast-track: 3 consecutive \u22650.9",
             fontsize=9, color=SLATE, ha="center", va="center")

    _save(fig, "tier_pyramid.png")


def dataset_composition() -> None:
    traj_labels = ["success", "continuation", "failure recovery",
                   "verification", "hint usage"]
    traj_sizes = [55, 20, 15, 5, 5]

    # Expert excluded entirely \u2014 0% is meaningless on a donut.
    tier_labels = ["warmup", "beginner", "intermediate", "advanced"]
    tier_sizes = [50, 30, 15, 5]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("SFT dataset composition  \u2022  1,500 rows",
                 fontsize=16, fontweight="bold", color=INK, y=1.02)
    fig.subplots_adjust(wspace=0.7, left=0.04, right=0.96)

    def donut(ax, sizes, labels, title, colors, center_label):
        wedges, _ = ax.pie(
            sizes, labels=None, colors=colors,
            wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
            startangle=90,
        )
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=10)
        legend_labels = [f"{l}  \u2014  {s}%" for l, s in zip(labels, sizes)]
        ax.legend(wedges, legend_labels, loc="center left",
                  bbox_to_anchor=(1.05, 0.5), frameon=False, fontsize=11)
        ax.text(0, 0, center_label, fontsize=14, fontweight="bold",
                color=INK, ha="center", va="center")

    donut(axes[0], traj_sizes, traj_labels, "Trajectory types",
          ["#22c55e", "#3a86ff", "#fb5607", "#8338ec", "#ffbe0b"],
          "5 types")
    donut(axes[1], tier_sizes, tier_labels, "Tier weights",
          ["#22c55e", "#06b6d4", PALETTE[0], PALETTE[1]],
          "4 tiers\n+ expert*")

    fig.text(
        0.5, -0.04,
        "* expert tasks excluded from SFT (randomized state checks \u2192 no canonical script). "
        "GRPO handles them via live reward signal.",
        fontsize=10, color=SLATE, ha="center", style="italic",
    )

    _save(fig, "dataset_composition.png")


def reward_components() -> None:
    components = [
        ("task achieved",       1.00, "+", "achieve"),
        ("chaos survival",      0.05, "+", "achieve"),
        ("partial progress",    0.80, "+", "shape"),
        ("progress delta",      0.10, "+", "shape"),
        ("idempotent retry",    0.02, "+", "shape"),
        ("rollback (per pair)", 0.10, "-", "penalty"),
        ("command failed",      0.50, "-", "penalty"),
        ("hint decay (n=3)",    0.39, "-", "penalty"),
    ]
    color_map = {
        "achieve": "#22c55e",
        "shape":   PALETTE[0],
        "penalty": PINK,
    }

    labels = [c[0] for c in components]
    values = [c[1] if c[2] == "+" else -c[1] for c in components]
    colors = [color_map[c[3]] for c in components]
    signed = [f"{c[2]}{c[1]:.2f}" for c in components]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=1.5,
            height=0.72, alpha=0.92)

    for y, v, txt in zip(y_pos, values, signed):
        offset = 0.025 if v >= 0 else -0.025
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, y, txt, va="center", ha=ha,
                fontsize=11, color=INK, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11.5, color=INK)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlim(-0.65, 1.18)
    ax.set_xlabel("contribution to reward", fontsize=10.5, color=SLATE)
    ax.set_title("Reward shaping: every modifier the agent can earn or lose",
                 fontsize=14, fontweight="bold", color=INK, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=SLATE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color="#22c55e", label="achievement (full reward)"),
        mpatches.Patch(color=PALETTE[0], label="dense shaping signal"),
        mpatches.Patch(color=PINK,       label="penalty / decay"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=10)

    fig.text(
        0.5, -0.04,
        "Final reward is clamped to [0.0, 0.99] before completion (1.0 reserved for "
        "verified achievement). Hint decay applied last as a multiplier (0.85^n).",
        fontsize=9.5, color=SLATE, ha="center", style="italic",
    )

    _save(fig, "reward_components.png")


def main() -> None:
    hero()
    tier_pyramid()
    dataset_composition()
    reward_components()


if __name__ == "__main__":
    main()
