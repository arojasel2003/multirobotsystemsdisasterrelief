"""
results.py
----------
Multi-trial aggregation and comparison plots for Greedy vs CBBA.

Reads:
    results_{algo}_{mode}_seed{N}.json   (one file per seed per combo)

Outputs (one per mode where at least one algo has results):
    comparison_baseline.png
    comparison_stress.png
    comparison_scale8.png

If only one algo has data for a mode, a note is printed and that figure
is skipped entirely — a one-algo bar chart would be misleading as a
"comparison" figure.  The summary table still shows whatever data is available.

Usage:
    python results.py          # run from supervisor_controller/ directory
"""

import glob
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

ALGOS   = ["greedy", "cbba"]
MODES   = ["baseline", "stress", "scale8"]

METRIC_KEYS = [
    "task_completion_rate_%",
    "avg_completion_time_steps",
    "avg_battery_per_task_%",
    "reallocation_count",
    "tasks_expired",
    "robot_battery_failures",
    "time_lost_reallocation",
]

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_trials(algo, mode):
    """Load all seeds for an (algo, mode) combo; return mean + std per metric.

    Returns None when no matching files are found.
    """
    pattern = f"results_{algo}_{mode}_seed*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    runs = [json.load(open(f)) for f in files]  # noqa: SIM115 — simple read

    out = {"algorithm": algo, "mode": mode, "n_trials": len(runs)}
    for k in METRIC_KEYS:
        vals = [r.get(k, 0) for r in runs]
        out[f"{k}_mean"] = statistics.mean(vals)
        out[f"{k}_std"]  = statistics.stdev(vals) if len(vals) > 1 else 0.0

    # Pool raw lists across all seeds for the histogram
    out["completion_times_all"] = [t for r in runs for t in r.get("completion_times", [])]
    out["battery_per_task_all"] = [b for r in runs for b in r.get("battery_per_task", [])]
    return out

# ─── STYLE ────────────────────────────────────────────────────────────────────

RED      = "#E74C3C"
GREEN    = "#2ECC71"
PANEL_BG = "#16213e"
FIG_BG   = "#1a1a2e"
TICK_COL = "#aaaacc"
LBL_COL  = "#ccccee"


def style_ax(ax, title, ylabel):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=LBL_COL)
    ax.tick_params(colors=TICK_COL, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.yaxis.label.set_color(LBL_COL)
    ax.yaxis.grid(True, color="#333355", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def bar_pair(ax, values, title, ylabel, fmt="{:.1f}", yerr=None):
    """Draw a two-bar grouped chart for Greedy vs CBBA.

    Parameters
    ----------
    values : [greedy_mean, cbba_mean]
    yerr   : [greedy_std,  cbba_std]  — optional; adds error caps when provided
    """
    error_kw = dict(elinewidth=1.2, ecolor="#ffffff88", capsize=5) if yerr else {}
    bars = ax.bar(
        ["Greedy", "CBBA"], values,
        color=[RED, GREEN], width=0.45,
        edgecolor="#ffffff33", linewidth=0.8, zorder=3,
        yerr=yerr, error_kw=error_kw,
    )
    mx = max(values) if max(values) > 0 else 1
    ax.set_ylim(0, mx * 1.35)  # slightly taller to leave room for error caps
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + mx * 0.04,
            fmt.format(val),
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="white",
        )
    style_ax(ax, title, ylabel)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Greedy", "CBBA"], fontsize=10, color="white")


# ─── FIGURE RENDERING ─────────────────────────────────────────────────────────

def render_figure(mode, greedy_agg, cbba_agg):
    """Produce comparison_{mode}.png from two aggregated result dicts."""

    def mean(agg, key):
        return agg[f"{key}_mean"] if agg else 0.0

    def std(agg, key):
        return agg[f"{key}_std"] if agg else 0.0

    fig = plt.figure(figsize=(18, 11), facecolor=FIG_BG)
    mode_label = mode.upper()
    fig.suptitle(
        f"Multi-Robot Disaster Relief — Greedy vs CBBA  [{mode_label}]",
        fontsize=18, fontweight="bold", color="white",
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97,
        top=0.90, bottom=0.07,
    )
    ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    # Plot 1: Task Completion Rate
    bar_pair(
        ax[0][0],
        [mean(greedy_agg, "task_completion_rate_%"),
         mean(cbba_agg,   "task_completion_rate_%")],
        "Task Completion Rate", "% Tasks Completed", "{:.1f}%",
        yerr=[std(greedy_agg, "task_completion_rate_%"),
              std(cbba_agg,   "task_completion_rate_%")],
    )
    ax[0][0].set_ylim(0, 130)

    # Plot 2: Avg Task Completion Time
    bar_pair(
        ax[0][1],
        [mean(greedy_agg, "avg_completion_time_steps"),
         mean(cbba_agg,   "avg_completion_time_steps")],
        "Avg Task Completion Time", "Timesteps", "{:.1f}",
        yerr=[std(greedy_agg, "avg_completion_time_steps"),
              std(cbba_agg,   "avg_completion_time_steps")],
    )

    # Plot 3: Reallocation Event Count
    realloc_vals = [mean(greedy_agg, "reallocation_count"),
                    mean(cbba_agg,   "reallocation_count")]
    bar_pair(
        ax[0][2],
        realloc_vals,
        "Reallocation Event Count", "Number of Reallocations", "{:.0f}",
        yerr=[std(greedy_agg, "reallocation_count"),
              std(cbba_agg,   "reallocation_count")],
    )
    ax[0][2].set_ylim(0, max(max(realloc_vals), 1) * 1.6)

    # Plot 4: Avg Battery Consumption per Task
    bar_pair(
        ax[1][0],
        [mean(greedy_agg, "avg_battery_per_task_%"),
         mean(cbba_agg,   "avg_battery_per_task_%")],
        "Avg Battery per Task", "Battery % Used", "{:.2f}%",
        yerr=[std(greedy_agg, "avg_battery_per_task_%"),
              std(cbba_agg,   "avg_battery_per_task_%")],
    )

    # Plot 5: Time Lost to Reallocation
    time_lost_vals = [mean(greedy_agg, "time_lost_reallocation"),
                      mean(cbba_agg,   "time_lost_reallocation")]
    bar_pair(
        ax[1][1],
        time_lost_vals,
        "Time Lost to Reallocation", "Timesteps Lost", "{:.0f}",
        yerr=[std(greedy_agg, "time_lost_reallocation"),
              std(cbba_agg,   "time_lost_reallocation")],
    )
    ax[1][1].set_ylim(0, max(max(time_lost_vals), 1) * 1.6)

    # Plot 6: Completion Time Distribution (pooled across seeds)
    ax6 = ax[1][2]
    greedy_times = greedy_agg["completion_times_all"] if greedy_agg else []
    cbba_times   = cbba_agg["completion_times_all"]   if cbba_agg   else []
    all_times    = greedy_times + cbba_times
    bins = np.linspace(0, max(all_times) if all_times else 1, 16)

    if greedy_times:
        ax6.hist(greedy_times, bins=bins, alpha=0.72,
                 color=RED,   label="Greedy", edgecolor="#ffffff44", zorder=3)
    if cbba_times:
        ax6.hist(cbba_times,   bins=bins, alpha=0.72,
                 color=GREEN, label="CBBA",   edgecolor="#ffffff44", zorder=3)

    style_ax(ax6, "Completion Time Distribution (pooled)", "Frequency")
    ax6.set_xlabel("Timesteps to Complete", fontsize=9, color=LBL_COL)
    ax6.legend(fontsize=9, facecolor="#0f0f23",
               edgecolor="#555577", labelcolor="white")

    out_path = f"comparison_{mode}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Results] Saved {out_path}")


# ─── SUMMARY TABLE ────────────────────────────────────────────────────────────

METRIC_LABELS = [
    ("Task Completion Rate (%)",     "task_completion_rate_%"),
    ("Avg Completion Time (steps)",  "avg_completion_time_steps"),
    ("Avg Battery per Task (%)",     "avg_battery_per_task_%"),
    ("Reallocation Count",           "reallocation_count"),
    ("Tasks Expired",                "tasks_expired"),
    ("Battery Failures",             "robot_battery_failures"),
    ("Time Lost to Realloc (steps)", "time_lost_reallocation"),
]

COL_W = 20  # width for each mean±std column


def fmt_cell(agg, key):
    if agg is None:
        return "  n/a".rjust(COL_W)
    mean_v = agg[f"{key}_mean"]
    std_v  = agg[f"{key}_std"]
    return f"{mean_v:.1f} ± {std_v:.1f}".rjust(COL_W)


def print_summary_table(aggregates):
    """aggregates: dict of {(algo, mode): agg_dict | None}"""
    header_w = 33
    sep = "=" * (header_w + 2 + COL_W * 2 + 4)

    for mode in MODES:
        g_agg = aggregates.get(("greedy", mode))
        c_agg = aggregates.get(("cbba",   mode))
        if g_agg is None and c_agg is None:
            continue  # already printed [skip] above

        n_g = g_agg["n_trials"] if g_agg else 0
        n_c = c_agg["n_trials"] if c_agg else 0

        print(f"\n{sep}")
        mode_title = f"MODE: {mode.upper()}  (greedy n={n_g}, cbba n={n_c})"
        print(f"  {mode_title}")
        print(sep)
        print(f"  {'METRIC':<{header_w}} {'GREEDY mean±std':>{COL_W}} {'CBBA mean±std':>{COL_W}}")
        print("-" * (header_w + 2 + COL_W * 2 + 4))
        for label, key in METRIC_LABELS:
            g_cell = fmt_cell(g_agg, key)
            c_cell = fmt_cell(c_agg, key)
            print(f"  {label:<{header_w}} {g_cell} {c_cell}")
    print(f"\n{'=' * (header_w + 2 + COL_W * 2 + 4)}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Change to the directory of this script so glob patterns resolve correctly
    # regardless of where the user invokes Python from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    aggregates = {}
    any_data = False

    for mode in MODES:
        g_agg = load_trials("greedy", mode)
        c_agg = load_trials("cbba",   mode)

        aggregates[("greedy", mode)] = g_agg
        aggregates[("cbba",   mode)] = c_agg

        if g_agg is None and c_agg is None:
            print(f"[skip] no results for {mode}")
            continue

        if g_agg is None:
            print(f"[skip] no results for greedy/{mode} — skipping comparison figure")
        if c_agg is None:
            print(f"[skip] no results for cbba/{mode} — skipping comparison figure")

        # Only render a comparison figure when both algos have data.
        # A one-algo chart would be misleading as a "comparison" output.
        if g_agg is not None and c_agg is not None:
            render_figure(mode, g_agg, c_agg)
            any_data = True
        else:
            any_data = True  # at least one algo has data; still show in table

    if not any_data:
        print("\nNo multi-seed result files found. Run the simulations first.")
        print("Expected pattern: results_{algo}_{mode}_seed{N}.json")
        return

    print_summary_table(aggregates)


if __name__ == "__main__":
    main()
