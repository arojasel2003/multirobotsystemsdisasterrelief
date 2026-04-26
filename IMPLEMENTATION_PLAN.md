# Implementation Plan — Multi-Robot Disaster Relief

Handoff doc for agent to execute remaining technical work before final submission.

Source: midterm report §VI (Future Plan) + grader rubric notes.

---

## Repo layout

```
controllers/
  robot_controller/robot_controller.py   # per-robot battery/state reporter
  supervisor_controller/
    supervisor_controller.py             # main loop + triggers + assignment
    astar.py                             # path planning, occupancy grid
    greedy.py                            # baseline allocator
    cbba.py                              # CBBA allocator
    results.py                           # post-sim plotting
    results_greedy.json                  # current baseline output
    results_cbba.json                    # current baseline output
    comparison_plots.png                 # current comparison
worlds/
  disaster_relief final.wbt              # Webots world
```

Algorithm switch lives at `supervisor_controller.py:46` — `USE_CBBA = True/False`.

Output JSON naming convention going forward:
`results_{algorithm}_{mode}_seed{N}.json`
where `algorithm ∈ {greedy, cbba}`, `mode ∈ {baseline, stress, scale8}`, `N ∈ {1,2,3}`.

---

## Task 1 — Path planning fix

### Problem
Robots whose Webots translation lands very close to an obstacle boundary get `world_to_grid` results that occasionally snap into an obstacle cell. The BFS fallback in `astar._find_nearest_free` recovers, but the resulting path includes a backout step. Report §VI ¶3 calls this out.

### Files
- `controllers/supervisor_controller/astar.py`
- `controllers/supervisor_controller/supervisor_controller.py`

### Changes

**A. `astar.py` — add helper.**

Add new public function below `grid_to_world`:

```python
def snap_to_cell_center(x, z):
    """
    Snap a world (x, z) coordinate to the centre of the nearest free grid
    cell. If the natural cell is free, returns its centre. Otherwise BFS
    outward to the closest free cell and returns that centre.
    """
    raw_row, raw_col = world_to_grid(x, z)
    free_row, free_col = _find_nearest_free(raw_row, raw_col)
    return grid_to_world(free_row, free_col)
```

Do NOT change `OCCUPANCY_GRID` or add inflation. Inflation risks disconnecting the 10×10 grid (boxes some cells off entirely). Cell-center snapping plus the existing corner-cutting prevention in `get_neighbours` and the runtime `is_cell_free` check in `move_step` already cover safety.

**B. `supervisor_controller.py` — use snap.**

Import the helper:
```python
from astar import astar, simplify_path, OCCUPANCY_GRID, world_to_grid, grid_to_world, snap_to_cell_center
```

At init (replace lines 369–373):
```python
for def_name in ROBOT_DEFS:
    x, z = get_position(def_name)
    sx, sz = snap_to_cell_center(x, z)
    set_position(def_name, sx, sz)
    robot_states[def_name]["x"] = sx
    robot_states[def_name]["z"] = sz
    print(f"[Supervisor] {def_name} start snapped: ({x:.2f},{z:.2f}) → ({sx:.2f},{sz:.2f})")
```

In assignment block (around line 584), snap start before A*:
```python
sx, sz = snap_to_cell_center(rs["x"], rs["z"])
start = (sx, sz)
goal  = (task["x"], task["z"])
path  = simplify_path(astar(start, goal, rs["battery"]))
```

If the snap moved the robot, also `set_position(robot_name, sx, sz)` and update `rs["x"], rs["z"]` so the in-memory state stays consistent with Webots.

### Verification
- Run `python astar.py` standalone — confirms grid prints + sample path still valid.
- Run a baseline Webots sim. In console, look for `start snapped` lines. None should snap by more than 1 cell (~1m). If one snaps further, the start position in `.wbt` is in/very near an obstacle — flag for the user.
- Compare path lengths (waypoint counts) to a pre-fix run for the same task assignments. Should be equal or shorter.

---

## Task 2 — Adaptive priority weight in CBBA

### Problem
`PRIORITY_WEIGHT = 5.0` is constant (`cbba.py:25`). When many priority-3 tasks arrive at once, every robot bids huge on the same urgent set, leaving priority-1/2 tasks unattended for long stretches. Report §VI ¶4.

### Files
- `controllers/supervisor_controller/cbba.py`

### Changes

**A. New helper.** Add above `compute_score`:

```python
def compute_priority_pressure(robots, tasks):
    """
    Returns a multiplier in [0.5, 2.0] that scales PRIORITY_WEIGHT.
    pressure = unassigned_high_pri / max(1, available_robots)
    Low pressure (e.g. 1 hi-pri task, 5 robots) → multiplier ~0.5
      → priority matters less, distance/battery dominate, robots spread out.
    High pressure (e.g. 5 hi-pri tasks, 1 robot) → multiplier 2.0
      → priority dominates, urgent tasks get hit first.
    """
    high_pri = sum(1 for t in tasks
                   if t["priority"] == 3
                   and not t["assigned"]
                   and not t.get("done", False))
    available = sum(1 for r in robots if is_available(r))
    if available == 0:
        return 1.0
    raw = high_pri / available
    return max(0.5, min(2.0, raw))
```

**B. Thread pressure through.** Update `compute_score` signature:

```python
def compute_score(robot, task, bundle, priority_pressure=1.0):
    dist             = distance(robot["x"], robot["z"], task["x"], task["z"])
    battery_factor   = (robot["battery"] / 100.0) * BATTERY_WEIGHT
    priority_factor  = task["priority"] * PRIORITY_WEIGHT * priority_pressure
    distance_penalty = dist * DISTANCE_WEIGHT
    bundle_penalty   = len(bundle) * 0.5
    return priority_factor + battery_factor - distance_penalty - bundle_penalty
```

Update `build_bundles` signature: `def build_bundles(robots, tasks, priority_pressure)`. Pass `priority_pressure` into both `compute_score` calls inside (the main loop and the fallback closest-task call).

In `cbba_allocate`, before calling `build_bundles`:

```python
priority_pressure = compute_priority_pressure(available_robots, unassigned_tasks)
print(f"[CBBA] Priority pressure: {priority_pressure:.2f}")
bundles, bids = build_bundles(available_robots, unassigned_tasks, priority_pressure)
```

### Verification
Add to `__main__` block in `cbba.py`:

```python
# Low-pressure scenario: 1 hi-pri, 5 robots → pressure ≈ 0.5
low_pressure_tasks = [
    {"id": "t_hi", "x": 0, "z": 0, "priority": 3, "assigned": False, "done": False, "assigned_to": None},
    {"id": "t_lo1", "x": -4, "z": -4, "priority": 1, "assigned": False, "done": False, "assigned_to": None},
    {"id": "t_lo2", "x":  4, "z":  4, "priority": 1, "assigned": False, "done": False, "assigned_to": None},
]
print("\n--- Low pressure test ---")
print(cbba_allocate(test_robots, low_pressure_tasks))

# High-pressure scenario: 5 hi-pri, 5 robots → pressure ≈ 1.0
# (still scales sensibly — all robots split evenly)
high_pressure_tasks = [
    {"id": f"t_{i}", "x": (i-2)*1.5, "z": 0, "priority": 3,
     "assigned": False, "done": False, "assigned_to": None}
    for i in range(5)
]
print("\n--- High pressure test ---")
print(cbba_allocate(test_robots, high_pressure_tasks))
```

Expected: low-pressure test → robots split across hi+lo tasks. High-pressure → robots distribute across the hi-pri set rather than all bidding on one.

---

## Task 3 — Stress mode for reallocation

### Problem
Current run logs 0 reallocation events (table I in report). Three triggers (battery / stuck / priority) are wired but never fire. Report §VI ¶1 names this the most immediate priority.

### Files
- `controllers/supervisor_controller/supervisor_controller.py`

### Changes

**A. Mode flag.** Below `USE_CBBA` near line 46:

```python
STRESS_MODE = False   # True = aggressive battery drain + faster spawn + forced stuck
SCALE_MODE  = False   # True = 8 robots (see Task 4)
SEED        = 1       # fixed seed for reproducibility (see Task 5)

random.seed(SEED)

MODE_TAG = ("stress" if STRESS_MODE
            else "scale8" if SCALE_MODE
            else "baseline")
```

**B. Tunables react to mode.** Replace the constants block (lines 67–73):

```python
BATTERY_CRITICAL    = 30.0
STUCK_LIMIT         = 200
TASK_DEADLINE       = 3000
MAX_SPEED           = 1.5
ARRIVAL_THRESH      = 0.5
CHARGING_STATION    = (4.26, 4.5)

if STRESS_MODE:
    TASK_SPAWN_INTERVAL = 150
    BATTERY_DRAIN_MOVE  = 0.05
    BATTERY_DRAIN_CHARGE_TRIP = 0.02
    FORCED_STUCK_ROBOT  = "robot_0"
    FORCED_STUCK_START  = 500
    FORCED_STUCK_END    = 500 + STUCK_LIMIT + 50  # release after trigger fires
else:
    TASK_SPAWN_INTERVAL = 500
    BATTERY_DRAIN_MOVE  = 0.008
    BATTERY_DRAIN_CHARGE_TRIP = 0.004
    FORCED_STUCK_ROBOT  = None
    FORCED_STUCK_START  = -1
    FORCED_STUCK_END    = -1
```

Update battery drain at lines 419–422 to use `BATTERY_DRAIN_MOVE` and `BATTERY_DRAIN_CHARGE_TRIP`.

**C. Forced stuck.** In `move_step` (line 195), at the top:

```python
def move_step(def_name, rs):
    # Stress mode: freeze designated robot during stuck window
    if (STRESS_MODE
            and def_name == FORCED_STUCK_ROBOT
            and FORCED_STUCK_START <= timestep_count <= FORCED_STUCK_END):
        return False
    ...
```

`timestep_count` is module-scope global from the main loop, so this works. If linting complains, pass it in or use `globals()`.

**D. Output filename.** Update `export_metrics` (line 355):

```python
filename = f"results_{ALGORITHM_NAME.lower()}_{MODE_TAG}_seed{SEED}.json"
```

### Verification
Set `STRESS_MODE = True`, run sim. Expect in console:
- `TRIGGER stuck: robot_0` between t=700 and t=750 (200 stuck steps after start at 500).
- `TRIGGER battery` from at least one robot before t=2000 (battery hits 30% with 0.05/step drain).
- `TRIGGER priority` whenever a spawned task rolls priority 3.

If no battery trigger fires by t=2000, raise `BATTERY_DRAIN_MOVE` to 0.08.
If stuck trigger doesn't fire, verify `timestep_count` is visible inside `move_step` — if not, refactor `move_step` to take it as an arg.

---

## Task 4 — Scalability mode (8 robots)

### Problem
Report §VI ¶2 commits to 8-robot trials to test whether CBBA's gain over Greedy holds as fleet grows.

### Files
- `worlds/disaster_relief final.wbt`
- `controllers/supervisor_controller/supervisor_controller.py`

### Changes

**A. Add 3 robots in `.wbt`.** Find the existing `robot_4` block. Duplicate it three times. Update each copy:

| DEF       | translation (x, y, z)     | Rationale (free cell verified vs `_RAW_GRID`) |
|-----------|---------------------------|------------------------------------------------|
| `robot_5` | `-3.5  <y>  3.5`          | row 8, col 1 — free                            |
| `robot_6` |  `3.5  <y> -3.5`          | row 1, col 8 — free                            |
| `robot_7` |  `0.5  <y>  0.5`          | row 5, col 5 — free                            |

Keep `<y>` (the height component) identical to `robot_4`. Each robot needs unique `controller "robot_controller"` field — already standard. Confirm `name "robot_5"` etc. matches DEF.

**B. Supervisor mode.**

```python
if SCALE_MODE:
    ROBOT_DEFS = ["robot_0", "robot_1", "robot_2", "robot_3", "robot_4",
                  "robot_5", "robot_6", "robot_7"]
else:
    ROBOT_DEFS = ["robot_0", "robot_1", "robot_2", "robot_3", "robot_4"]
```

`MODE_TAG` (Task 3) handles output naming.

**C. Sanity check.** When `SCALE_MODE = True` AND a robot DEF is missing from the `.wbt` (loaded `getFromDef` returns None), abort with a clear print. Otherwise robot just sits at (0,0) and tanks the metrics silently.

### Verification
- Run with `SCALE_MODE = True, USE_CBBA = False` once → confirm all 8 robots show in init prints with sane positions.
- Look at `[CBBA] Allocating: 8 robots, N tasks` in CBBA mode log.
- Output goes to `results_{algo}_scale8_seed1.json`.

---

## Task 5 — Seeded multi-trial runs + aggregated plots

### Problem
Single-run numbers ≠ statistically meaningful. Report §VI ¶5 commits to 3 seeded trials per condition.

### Run matrix

| algo   | mode     | seeds      | output files                                              |
|--------|----------|------------|-----------------------------------------------------------|
| greedy | baseline | 1, 2, 3    | `results_greedy_baseline_seed{1,2,3}.json`                |
| cbba   | baseline | 1, 2, 3    | `results_cbba_baseline_seed{1,2,3}.json`                  |
| greedy | stress   | 1, 2, 3    | `results_greedy_stress_seed{1,2,3}.json`                  |
| cbba   | stress   | 1, 2, 3    | `results_cbba_stress_seed{1,2,3}.json`                    |
| greedy | scale8   | 1, 2, 3    | `results_greedy_scale8_seed{1,2,3}.json`                  |
| cbba   | scale8   | 1, 2, 3    | `results_cbba_scale8_seed{1,2,3}.json`                    |

Total: 18 Webots runs. Each is ~10,000 timesteps.

### Workflow

For each combination: edit `USE_CBBA`, `STRESS_MODE`, `SCALE_MODE`, `SEED` at top of `supervisor_controller.py`, run Webots, wait for `[Supervisor] Simulation complete.`, save JSON, repeat.

Optional helper: write `run_matrix.md` listing the 18 flag combos so a human can check them off.

### `results.py` rewrite

Replace single-pair load logic with:

```python
import glob, statistics

def load_trials(algo, mode):
    """Load all seeds for an (algo, mode) combo, return mean+std per metric."""
    pattern = f"results_{algo}_{mode}_seed*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    runs = [json.load(open(f)) for f in files]
    keys = ["task_completion_rate_%", "avg_completion_time_steps",
            "avg_battery_per_task_%", "reallocation_count",
            "tasks_expired", "robot_battery_failures",
            "time_lost_reallocation"]
    out = {"algorithm": algo, "mode": mode, "n_trials": len(runs)}
    for k in keys:
        vals = [r.get(k, 0) for r in runs]
        out[f"{k}_mean"] = statistics.mean(vals)
        out[f"{k}_std"]  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    out["completion_times_all"] = [t for r in runs for t in r["completion_times"]]
    out["battery_per_task_all"] = [b for r in runs for b in r["battery_per_task"]]
    return out
```

Render three figures (`comparison_baseline.png`, `comparison_stress.png`, `comparison_scale8.png`) using the existing `bar_pair` style but pass `yerr=std` to `ax.bar`. Histogram uses pooled `completion_times_all`.

Add a printed summary table grouped by mode showing mean ± std for each metric, both algos side-by-side.

### Verification
- Drop in 18 JSONs (or 6 — script should tolerate missing modes gracefully, prints `[skip] no results for {algo}/{mode}`).
- Confirm error bars render and std values are non-zero where expected.

---

## Suggested execution order

1. Task 1 (path snap). Code only. Smoke-test in Webots: 1 baseline run per algo. Sanity-check vs current `results_greedy.json` / `results_cbba.json` numbers — should be similar or slightly better.
2. Task 2 (adaptive priority). Code only. Re-run baseline (overwrites the smoke-tests from step 1 because seeded — that's fine).
3. Task 3 (stress mode). Code only. One stress run per algo to confirm triggers fire.
4. Task 4 (8 robots). World file edit + flag. One scale8 run per algo to confirm it loads.
5. Task 5 — formal run matrix (18 runs) + new `results.py`. Render all three comparison figures.

After step 5: hand back to the user for writeup.

---

## Out of scope for this plan

- Any changes to the report PDF / writeup (Introduction grounding, References section, figure font size, layout). User said report is not a concern right now.
- New algorithms beyond Greedy + CBBA.
- Webots world geometry changes beyond adding 3 robot DEFs.
- Robot controller changes — `robot_controller.py` is a thin state reporter and stays as-is.

---

## Notes / gotchas

- `OCCUPANCY_GRID` is indexed `[row][col]` where `row` corresponds to `z` and `col` to `x`. `world_to_grid(x, z)` returns `(row, col)` in that order. Easy to flip — verify any new grid lookups.
- `_FREE_CELLS` cache (supervisor lines 78–87) excludes the charging-station corner. New robot starts in Task 4 must not be in that excluded zone.
- CBBA `MAX_BUNDLE_SIZE = 3` — with 8 robots and few tasks, total bundle slots (24) far exceed task supply, so consensus phase becomes more important. Watch consensus iterations log; if it hits 100 (`MAX_ITERATIONS`) regularly, bump or investigate.
- Random seed must be set BEFORE any `random.choice` call. Currently the seed line goes near the top of supervisor — verify `_FREE_CELLS` precomputation does not consume randomness (it doesn't, but the dynamic spawn does).
- Forced stuck (Task 3) only freezes movement. Battery still drains and stuck counter increments — that's the point. Robot resumes movement after `FORCED_STUCK_END`, but by then it has already been reset to idle by the trigger handler.
