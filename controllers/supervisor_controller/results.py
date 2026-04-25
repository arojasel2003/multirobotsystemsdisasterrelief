"""
results.py
----------
Run this AFTER completing both simulation runs to generate
comparison plots between Greedy and CBBA algorithms.

Usage:
    python results.py

Reads:
    results_greedy.json
    results_cbba.json

Outputs:
    comparison_plots.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─── LOAD RESULTS ────────────────────────────────────────────────────────────

def load_results(filename):
    if not os.path.exists(filename):
        print(f"[Results] WARNING: {filename} not found. "
              f"Run the simulation first.")
        return None
    with open(filename) as f:
        return json.load(f)

greedy = load_results("results_greedy.json")
cbba   = load_results("results_cbba.json")

if not greedy and not cbba:
    print("No result files found. Run both simulations first.")
    exit()

# Use dummy data if one file is missing
if not greedy:
    greedy = {
        "algorithm": "Greedy",
        "task_completion_rate_%": 0,
        "avg_completion_time_steps": 0,
        "avg_battery_per_task_%": 0,
        "reallocation_count": 0,
        "tasks_expired": 0,
        "robot_battery_failures": 0,
        "time_lost_reallocation": 0,
        "completion_times": [],
        "battery_per_task": [],
    }

if not cbba:
    cbba = {
        "algorithm": "CBBA",
        "task_completion_rate_%": 0,
        "avg_completion_time_steps": 0,
        "avg_battery_per_task_%": 0,
        "reallocation_count": 0,
        "tasks_expired": 0,
        "robot_battery_failures": 0,
        "time_lost_reallocation": 0,
        "completion_times": [],
        "battery_per_task": [],
    }

# ─── STYLE ───────────────────────────────────────────────────────────────────

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

def bar_pair(ax, values, title, ylabel, fmt="{:.1f}"):
    bars = ax.bar(["Greedy", "CBBA"], values,
                  color=[RED, GREEN], width=0.45,
                  edgecolor="#ffffff33", linewidth=0.8, zorder=3)
    mx = max(values) if max(values) > 0 else 1
    ax.set_ylim(0, mx * 1.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + mx * 0.04,
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="white")
    style_ax(ax, title, ylabel)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Greedy", "CBBA"], fontsize=10, color="white")

# ─── FIGURE ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 11), facecolor=FIG_BG)
fig.suptitle("Multi-Robot Disaster Relief — Greedy vs CBBA",
             fontsize=18, fontweight="bold", color="white")

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97,
                       top=0.90, bottom=0.07)

ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

# ─── PLOT 1: Task Completion Rate ────────────────────────────────────────────

bar_pair(ax[0][0],
         [greedy["task_completion_rate_%"], cbba["task_completion_rate_%"]],
         "Task Completion Rate", "% Tasks Completed", "{:.1f}%")
ax[0][0].set_ylim(0, 115)

# ─── PLOT 2: Avg Task Completion Time ────────────────────────────────────────

bar_pair(ax[0][1],
         [greedy["avg_completion_time_steps"], cbba["avg_completion_time_steps"]],
         "Avg Task Completion Time", "Timesteps", "{:.1f}")

# ─── PLOT 3: Reallocation Event Count ────────────────────────────────────────

bar_pair(ax[0][2],
         [greedy["reallocation_count"], cbba["reallocation_count"]],
         "Reallocation Event Count", "Number of Reallocations", "{:.0f}")
ax[0][2].set_ylim(0, max(max(greedy["reallocation_count"],
                              cbba["reallocation_count"]), 1) * 1.5)

# ─── PLOT 4: Avg Battery Consumption per Task ────────────────────────────────

bar_pair(ax[1][0],
         [greedy["avg_battery_per_task_%"], cbba["avg_battery_per_task_%"]],
         "Avg Battery per Task", "Battery % Used", "{:.2f}%")

# ─── PLOT 5: Time Lost to Reallocation ───────────────────────────────────────

bar_pair(ax[1][1],
         [greedy["time_lost_reallocation"], cbba["time_lost_reallocation"]],
         "Time Lost to Reallocation", "Timesteps Lost", "{:.0f}")
ax[1][1].set_ylim(0, max(max(greedy["time_lost_reallocation"],
                              cbba["time_lost_reallocation"]), 1) * 1.5)

# ─── PLOT 6: Completion Time Distribution ────────────────────────────────────

ax6 = ax[1][2]
all_times = greedy["completion_times"] + cbba["completion_times"]
bins = np.linspace(0, max(all_times) if all_times else 1, 16)

if greedy["completion_times"]:
    ax6.hist(greedy["completion_times"], bins=bins, alpha=0.72,
             color=RED,   label="Greedy", edgecolor="#ffffff44", zorder=3)
if cbba["completion_times"]:
    ax6.hist(cbba["completion_times"],   bins=bins, alpha=0.72,
             color=GREEN, label="CBBA",   edgecolor="#ffffff44", zorder=3)

style_ax(ax6, "Completion Time Distribution", "Frequency")
ax6.set_xlabel("Timesteps to Complete", fontsize=9, color=LBL_COL)
ax6.legend(fontsize=9, facecolor="#0f0f23",
           edgecolor="#555577", labelcolor="white")

# ─── SAVE ────────────────────────────────────────────────────────────────────

plt.savefig("comparison_plots.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("[Results] Saved comparison_plots.png")

# ─── PRINT SUMMARY TABLE ─────────────────────────────────────────────────────

print("\n" + "=" * 55)
print(f"{'METRIC':<35} {'GREEDY':>8} {'CBBA':>8}")
print("=" * 55)
metrics_to_show = [
    ("Task Completion Rate (%)",     "task_completion_rate_%"),
    ("Avg Completion Time (steps)",  "avg_completion_time_steps"),
    ("Reallocation Count",           "reallocation_count"),
    ("Avg Battery per Task (%)",     "avg_battery_per_task_%"),
    ("Time Lost to Realloc (steps)", "time_lost_reallocation"),
    ("Battery Failures",             "robot_battery_failures"),
    ("Tasks Expired",                "tasks_expired"),
]
for label, key in metrics_to_show:
    gv = greedy.get(key, 0)
    cv = cbba.get(key, 0)
    print(f"  {label:<33} {gv:>8.1f} {cv:>8.1f}")
print("=" * 55)
