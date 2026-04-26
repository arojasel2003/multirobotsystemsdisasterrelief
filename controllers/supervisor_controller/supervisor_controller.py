"""
supervisor_controller.py
------------------------
The brain of the disaster relief simulation.

ALGORITHM SWITCH:
  Set USE_CBBA = True  to run CBBA
  Set USE_CBBA = False to run Greedy (baseline)

FIXES APPLIED:
  1. Robot z-position init bug: get_position() returns (x, z) tuple;
     startup now correctly unpacks both components (was reading pos[1]=Y height).
  2. allocate() now only runs when there are actually idle robots AND
     unassigned tasks — no more redundant calls every timestep.
  3. check_deadlines: expired tasks are now freed from their assigned robot
     too, not just skipped because assigned_to is not None.
  4. prev_pos cleared when stuck trigger fires so the reset robot doesn't
     immediately re-trigger stuck detection on its very next timestep.
  5. BATTERY_CRITICAL raised to 30% to match robot_controller and allocators.
  6. TASK_SPAWN_INTERVAL set to 500 timesteps (~25 real seconds at 32ms step)
     so tasks appear at a reasonable visible rate — not too fast, not too slow.
  7. Stuck trigger: robots sent back to idle (not abandoned at arbitrary pos)
     and prev_pos reset so stuck counter starts clean.
  8. Task locations verified free of inflated obstacles — initial tasks
     placed in cells confirmed free in the inflated occupancy grid.
  9. Dynamic task spawn now uses cell-centre snapping so spawned tasks
     always land in free grid cells, never inside obstacle-inflated zones.
  10. Reallocation: supervisor releases task BEFORE calling reallocate()
      so allocators don't need to search for and release tasks themselves.
  11. allocate() / reallocate() wrappers call the right function based on
      USE_CBBA and pass consistent arguments in all code paths.
  12. move_step: arrival threshold tuned to 0.5m (was 0.6m) to better match
      cell-centre waypoints and prevent robots "arriving" a cell early.
  13. MAX_SPEED capped so robots don't teleport across multiple grid cells
      per timestep (was 2.0 m/s which at 32ms = 0.064m/step — fine — but
      verified against ARRIVAL_THRESH so they can't skip waypoints).
"""

from controller import Supervisor
import json
import random
import math

# ─── ALGORITHM SWITCH ────────────────────────────────────────────────────────

USE_CBBA = True   # False = Greedy, True = CBBA

STRESS_MODE = False   # True = aggressive battery drain + faster spawn + forced stuck
SCALE_MODE  = False   # True = 8 robots (see Task 4)
SEED        = 1       # fixed seed for reproducibility (see Task 5)

random.seed(SEED)

MODE_TAG = ("stress" if STRESS_MODE
            else "scale8" if SCALE_MODE
            else "baseline")

if USE_CBBA:
    from cbba import cbba_allocate, cbba_reallocate
    ALGORITHM_NAME = "CBBA"
else:
    from greedy import greedy_allocate, greedy_reallocate
    ALGORITHM_NAME = "Greedy"

from astar import astar, simplify_path, OCCUPANCY_GRID, world_to_grid, grid_to_world, snap_to_cell_center

# ─── SETUP ───────────────────────────────────────────────────────────────────

supervisor = Supervisor()
timestep   = int(supervisor.getBasicTimeStep())

print(f"[Supervisor] Running with algorithm: {ALGORITHM_NAME}")

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

if SCALE_MODE:
    ROBOT_DEFS = ["robot_0", "robot_1", "robot_2", "robot_3", "robot_4",
                  "robot_5", "robot_6", "robot_7"]
else:
    ROBOT_DEFS = ["robot_0", "robot_1", "robot_2", "robot_3", "robot_4"]
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
    FORCED_STUCK_END    = 500 + STUCK_LIMIT + 50
else:
    TASK_SPAWN_INTERVAL = 500
    BATTERY_DRAIN_MOVE  = 0.008
    BATTERY_DRAIN_CHARGE_TRIP = 0.004
    FORCED_STUCK_ROBOT  = None
    FORCED_STUCK_START  = -1
    FORCED_STUCK_END    = -1

# ─── FREE CELL CACHE ─────────────────────────────────────────────────────────
# Pre-compute all free cells (after inflation) for fast random task spawning

_FREE_CELLS = []
for _r in range(10):
    for _c in range(10):
        if OCCUPANCY_GRID[_r][_c] == 0:
            wx, wz = grid_to_world(_r, _c)
            # Include boundary cells — world is exactly -5..+5
            if -4.5 <= wx <= 4.5 and -4.5 <= wz <= 4.5:
                # Exclude charging station area (top-right, 4.26, 4.5)
                if not (wx >= 3.5 and wz >= 3.5):
                    _FREE_CELLS.append((wx, wz))

# ─── METRICS ─────────────────────────────────────────────────────────────────

metrics = {
    "algorithm":              ALGORITHM_NAME,
    "tasks_completed":        0,
    "tasks_expired":          0,
    "tasks_total":            0,
    "reallocation_count":     0,
    "completion_times":       [],
    "battery_per_task":       [],
    "robot_battery_failures": 0,
    "time_lost_reallocation": 0,
}

# ─── TASK POOL ───────────────────────────────────────────────────────────────
# Initial task positions chosen to sit at grid cell centres that are free
# in the inflated occupancy grid (verified manually):

tasks = [
    {"id": "task_0", "x": -4.5, "z": -4.5, "priority": 3,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
    {"id": "task_1", "x":  0.5, "z": -4.5, "priority": 3,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
    {"id": "task_2", "x":  4.5, "z": -4.5, "priority": 2,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
    {"id": "task_3", "x": -4.5, "z":  4.5, "priority": 3,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
    {"id": "task_4", "x":  0.5, "z":  4.5, "priority": 2,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
    {"id": "task_5", "x":  2.5, "z":  4.5, "priority": 1,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
    {"id": "task_6", "x":  0.5, "z":  0.5, "priority": 2,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
    {"id": "task_7", "x": -4.5, "z":  0.5, "priority": 1,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
    {"id": "task_8", "x":  4.5, "z":  0.5, "priority": 2,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
    {"id": "task_9", "x":  0.5, "z":  4.5, "priority": 3,
     "assigned": False, "assigned_to": None,
     "spawn_time": 0, "deadline": TASK_DEADLINE, "done": False},
]
metrics["tasks_total"] = len(tasks)

# ─── ROBOT STATE ─────────────────────────────────────────────────────────────

robot_states = {
    def_name: {
        "name":          def_name,
        "x":             0.0,
        "z":             0.0,
        "battery":       100.0,
        "state":         "idle",
        "task_id":       None,
        "waypoints":     [],
        "stuck_counter": 0,
        "prev_pos":      None,  # cleared on stuck reset (Bug 5 fix)
        "battery_start": 100.0,
        "assign_time":   0,
    }
    for def_name in ROBOT_DEFS
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_node(def_name):
    return supervisor.getFromDef(def_name)


def get_position(def_name):
    """Return (x, z) world position of a robot node."""
    node = get_node(def_name)
    if node is None:
        return (0.0, 0.0)
    pos = node.getField("translation").getSFVec3f()
    return (pos[0], pos[2])   # pos[0]=X, pos[1]=Y(height), pos[2]=Z


def set_position(def_name, x, z):
    """Teleport robot to (x, z) keeping its current Y height."""
    node = get_node(def_name)
    if node is None:
        return
    y = node.getField("translation").getSFVec3f()[1]
    node.getField("translation").setSFVec3f([x, y, z])


def dist(x1, z1, x2, z2):
    return math.sqrt((x1 - x2)**2 + (z1 - z2)**2)


def is_cell_free(x, z):
    """Check the inflated occupancy grid — returns True if passable."""
    col = max(0, min(9, int((x + 5.0))))
    row = max(0, min(9, int((z + 5.0))))
    return OCCUPANCY_GRID[row][col] == 0


def move_step(def_name, rs):
    """
    Move robot one timestep toward its current waypoint.
    Checks inflated occupancy grid so robots never enter obstacle cells.
    Returns True if robot has arrived at the current waypoint.
    """
    # Stress mode: freeze designated robot during stuck window
    if (STRESS_MODE
            and def_name == FORCED_STUCK_ROBOT
            and FORCED_STUCK_START <= timestep_count <= FORCED_STUCK_END):
        return False

    if not rs["waypoints"]:
        return False

    tx, tz = rs["waypoints"][0]
    cx, cz = rs["x"], rs["z"]
    d = dist(cx, cz, tx, tz)

    if d < ARRIVAL_THRESH:
        return True   # arrived at this waypoint

    step = MAX_SPEED * (timestep / 1000.0)
    step = min(step, d)   # never overshoot
    nx = cx + (tx - cx) / d * step
    nz = cz + (tz - cz) / d * step

    # Only move if the destination cell is free (obstacle-inflation safety net)
    if is_cell_free(nx, nz):
        set_position(def_name, nx, nz)
        rs["x"] = nx
        rs["z"] = nz
    # If blocked, stay put — stuck counter will eventually trigger reallocation

    return False


def remove_task_node(task_id):
    node = supervisor.getFromDef(task_id)
    if node:
        node.remove()
        print(f"[Supervisor] Removed {task_id} from world.")


def _spawn_task_node(task):
    """Add a visible coloured cylinder to the Webots world for a task.
    Red = priority 3, Orange = priority 2, Yellow = priority 1."""
    color = ("1 0 0" if task["priority"] == 3
             else ("1 0.5 0" if task["priority"] == 2 else "1 1 0"))
    supervisor.getRoot().getField("children").importMFNodeFromString(
        -1,
        f'DEF {task["id"]} Solid {{'
        f'  translation {task["x"]} 0.3 {task["z"]}'
        f'  children ['
        f'    Shape {{'
        f'      appearance PBRAppearance {{ baseColor {color} }}'
        f'      geometry Cylinder {{ radius 0.25 height 0.5 }}'
        f'    }}'
        f'  ]'
        f'  name "{task["id"]}"'
        f'}}'
    )


def spawn_dynamic_task(timestep_count):
    """
    Spawn a new dynamic task at a random free cell centre.
    Uses the pre-computed _FREE_CELLS list so tasks always land in valid spots.
    Returns a priority trigger dict if priority==3, else None.
    """
    if not _FREE_CELLS:
        return None

    x, z = random.choice(_FREE_CELLS)
    priority = random.choice([1, 2, 2, 3, 3])
    task_id  = f"dyn_{timestep_count}"

    new_task = {
        "id":          task_id,
        "x":           x,
        "z":           z,
        "priority":    priority,
        "assigned":    False,
        "assigned_to": None,
        "spawn_time":  timestep_count,
        "deadline":    timestep_count + TASK_DEADLINE,
        "done":        False,
    }
    tasks.append(new_task)
    metrics["tasks_total"] += 1

    _spawn_task_node(new_task)

    print(f"[Supervisor] Spawned {task_id} at ({x:.2f},{z:.2f}) "
          f"priority={priority} deadline={new_task['deadline']}")

    if priority == 3:
        return {"type": "priority", "task": task_id}
    return None


def check_deadlines(timestep_count):
    """
    Expire tasks that have exceeded their deadline.
    FIX: also frees the robot carrying an expired task (previously only
    unassigned tasks were checked — assigned ones were silently skipped).
    """
    for task in tasks:
        if task["done"]:
            continue
        if timestep_count > task["deadline"]:
            print(f"[Supervisor] Task {task['id']} EXPIRED at t={timestep_count}")
            task["done"]        = True
            task["assigned"]    = True   # marks as handled
            task["assigned_to"] = None
            metrics["tasks_expired"] += 1
            remove_task_node(task["id"])

            # Free any robot that was heading for this task
            for rs in robot_states.values():
                if rs["task_id"] == task["id"]:
                    rs["task_id"]   = None
                    rs["waypoints"] = []
                    rs["state"]     = "idle"
                    rs["prev_pos"]  = None
                    print(f"[Supervisor] Freed {rs['name']} from expired task.")


def allocate(robot_list, task_list):
    if USE_CBBA:
        return cbba_allocate(robot_list, task_list)
    else:
        return greedy_allocate(robot_list, task_list)


def reallocate(robot_list, trigger):
    """
    The supervisor has already updated task + robot state before calling here,
    so we pass the full tasks list and let the allocator pick freely.
    """
    if USE_CBBA:
        return cbba_reallocate(robot_list, tasks, trigger)
    else:
        return greedy_reallocate(robot_list, tasks, trigger)


def export_metrics():
    avg_completion = (
        sum(metrics["completion_times"]) / len(metrics["completion_times"])
        if metrics["completion_times"] else 0
    )
    avg_battery = (
        sum(metrics["battery_per_task"]) / len(metrics["battery_per_task"])
        if metrics["battery_per_task"] else 0
    )
    total     = metrics["tasks_total"]
    completed = metrics["tasks_completed"]
    rate      = (completed / total * 100) if total > 0 else 0

    summary = {
        **metrics,
        "task_completion_rate_%":    round(rate, 2),
        "avg_completion_time_steps": round(avg_completion, 2),
        "avg_battery_per_task_%":    round(avg_battery, 2),
    }

    filename = f"results_{ALGORITHM_NAME.lower()}_{MODE_TAG}_seed{SEED}.json"
    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Supervisor] Metrics exported to {filename}")
    print(json.dumps(summary, indent=2))

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

timestep_count  = 0
next_spawn_time = TASK_SPAWN_INTERVAL
realloc_triggers = []

# ── Init robot positions (FIX 1: unpack tuple correctly) ─────────────────────
for def_name in ROBOT_DEFS:
    # Sanity check: abort if a robot DEF is missing from the .wbt (scale mode guard)
    if SCALE_MODE and supervisor.getFromDef(def_name) is None:
        print(f"[Supervisor] FATAL: SCALE_MODE=True but DEF '{def_name}' not found "
              f"in the .wbt file. Add the robot node before running scale mode.")
        raise SystemExit(1)
    x, z = get_position(def_name)
    sx, sz = snap_to_cell_center(x, z)
    set_position(def_name, sx, sz)
    robot_states[def_name]["x"] = sx
    robot_states[def_name]["z"] = sz
    print(f"[Supervisor] {def_name} start snapped: ({x:.2f},{z:.2f}) → ({sx:.2f},{sz:.2f})")

# ── Remove any task nodes left over in the .wbt file ─────────────────────────
# The .wbt may contain hardcoded task DEFs (e.g. dynamic_task_1000) from a
# previous run. Remove them now so we start with a clean slate and no DEF
# name conflicts when spawning fresh nodes.
_STALE_DEFS = ["dynamic_task_1000"]  # add any other known stale names here
for _stale in _STALE_DEFS:
    _node = supervisor.getFromDef(_stale)
    if _node:
        _node.remove()
        print(f"[Supervisor] Removed stale node: {_stale}")

for _t in tasks:
    _spawn_task_node(_t)
    print(f"[Supervisor] Spawned initial task node: {_t['id']} at ({_t['x']}, {_t['z']}) priority={_t['priority']}")

print(f"[Supervisor] Starting — Algorithm: {ALGORITHM_NAME}")
print(f"[Supervisor] Robots: {ROBOT_DEFS}")
print(f"[Supervisor] Initial tasks: {len(tasks)}")
print(f"[Supervisor] Free spawn cells: {len(_FREE_CELLS)}")

while supervisor.step(timestep) != -1:
    timestep_count += 1

    # ── 1. Move all active robots one step ───────────────────────────────────
    for def_name in ROBOT_DEFS:
        rs = robot_states[def_name]

        # Safety net A: moving but no waypoints and no task → idle
        if rs["state"] == "moving" and not rs["waypoints"] and rs["task_id"] is None:
            rs["state"] = "idle"

        # Safety net B: robot's task was marked done externally → idle
        if rs["state"] == "moving" and rs["task_id"]:
            t = next((t for t in tasks if t["id"] == rs["task_id"]), None)
            if t and t["done"]:
                rs["task_id"]   = None
                rs["waypoints"] = []
                rs["state"]     = "idle"
                rs["prev_pos"]  = None

        if rs["state"] in ("moving", "moving_to_charge") and rs["waypoints"]:
            arrived = move_step(def_name, rs)

            # Battery drain
            if rs["state"] == "moving":
                rs["battery"] = max(0.0, rs["battery"] - BATTERY_DRAIN_MOVE)
            else:  # moving_to_charge
                rs["battery"] = max(0.0, rs["battery"] - BATTERY_DRAIN_CHARGE_TRIP)

            if arrived:
                rs["waypoints"].pop(0)

                if rs["waypoints"]:
                    pass   # more waypoints — keep moving

                else:
                    # No more waypoints — destination reached
                    if rs["state"] == "moving":
                        task_id      = rs["task_id"]
                        duration     = timestep_count - rs["assign_time"]
                        battery_used = rs["battery_start"] - rs["battery"]

                        print(f"[Supervisor] {def_name} completed {task_id} "
                              f"in {duration} steps, "
                              f"used {battery_used:.1f}% battery")

                        metrics["tasks_completed"]    += 1
                        metrics["completion_times"].append(duration)
                        metrics["battery_per_task"].append(max(0.0, battery_used))

                        for t in tasks:
                            if t["id"] == task_id:
                                t["done"]        = True
                                t["assigned"]    = True
                                t["assigned_to"] = None
                                break

                        remove_task_node(task_id)
                        rs["task_id"]  = None
                        rs["state"]    = "idle"
                        rs["prev_pos"] = None

                    elif rs["state"] == "moving_to_charge":
                        print(f"[Supervisor] {def_name} reached charging station.")
                        rs["state"] = "charging"

        elif rs["state"] == "charging":
            rs["battery"] = min(100.0, rs["battery"] + 1.0)
            if rs["battery"] >= 100.0:
                print(f"[Supervisor] {def_name} fully charged!")
                rs["state"] = "idle"

    # ── 2. Check deadlines ────────────────────────────────────────────────────
    check_deadlines(timestep_count)

    # ── 3. Check reallocation triggers ───────────────────────────────────────
    for def_name in ROBOT_DEFS:
        rs = robot_states[def_name]

        # Trigger 1: Battery critical
        if (rs["battery"] <= BATTERY_CRITICAL
                and rs["state"] not in ("charging", "moving_to_charge")
                and rs["task_id"] is not None):
            print(f"[Supervisor] TRIGGER battery: {def_name} "
                  f"({rs['battery']:.1f}%)")
            metrics["reallocation_count"]     += 1
            metrics["robot_battery_failures"] += 1

            # Release the task BEFORE calling reallocate
            for t in tasks:
                if t["id"] == rs["task_id"]:
                    t["assigned"]    = False
                    t["assigned_to"] = None
                    break

            realloc_triggers.append({"type": "battery", "robot": def_name})

            rs["task_id"]   = None
            rs["waypoints"] = [CHARGING_STATION]
            rs["state"]     = "moving_to_charge"
            rs["prev_pos"]  = None

        # Trigger 2: Stuck detection
        prev = rs["prev_pos"]
        if prev and rs["state"] == "moving":
            moved = dist(rs["x"], rs["z"], prev[0], prev[1])
            if moved < 0.005:
                rs["stuck_counter"] += 1
            else:
                rs["stuck_counter"] = 0
        elif rs["state"] != "moving":
            rs["stuck_counter"] = 0

        if rs["stuck_counter"] >= STUCK_LIMIT and rs["task_id"]:
            print(f"[Supervisor] TRIGGER stuck: {def_name} "
                  f"(no movement for {rs['stuck_counter']} steps)")
            metrics["reallocation_count"]     += 1
            metrics["time_lost_reallocation"] += rs["stuck_counter"]

            # Release the task BEFORE calling reallocate
            for t in tasks:
                if t["id"] == rs["task_id"]:
                    t["assigned"]    = False
                    t["assigned_to"] = None
                    break

            realloc_triggers.append({"type": "stuck", "robot": def_name})

            rs["task_id"]      = None
            rs["waypoints"]    = []
            rs["state"]        = "idle"
            rs["stuck_counter"] = 0
            rs["prev_pos"]     = None   # FIX 4: clear so counter starts fresh

        else:
            rs["prev_pos"] = (rs["x"], rs["z"])

    # ── 4. Dynamic task spawning ──────────────────────────────────────────────
    if timestep_count >= next_spawn_time:
        trigger = spawn_dynamic_task(timestep_count)
        next_spawn_time = timestep_count + TASK_SPAWN_INTERVAL
        if trigger:
            realloc_triggers.append(trigger)

    # ── 5. Run allocation ─────────────────────────────────────────────────────
    robot_list = list(robot_states.values())

    # FIX 2: Process realloc triggers
    if realloc_triggers:
        new_assignments = []
        for trigger in realloc_triggers:
            new_assignments += reallocate(robot_list, trigger)
        realloc_triggers.clear()

        # Deduplicate: if the same robot appears twice, keep only first
        seen_robots = set()
        deduped = []
        for rname, tid in new_assignments:
            if rname not in seen_robots:
                deduped.append((rname, tid))
                seen_robots.add(rname)
        new_assignments = deduped

    else:
        # FIX 2: Only allocate when there's something to do
        task_list = [t for t in tasks if not t["assigned"] and not t["done"]]
        has_idle  = any(
            rs["state"] == "idle" and rs["battery"] > 30.0
            for rs in robot_states.values()
        )
        if has_idle and task_list:
            new_assignments = allocate(robot_list, task_list)
        else:
            new_assignments = []

    # ── 6. Apply assignments — plan A* path and start moving ─────────────────
    for robot_name, task_id in new_assignments:
        rs   = robot_states.get(robot_name)
        task = next((t for t in tasks if t["id"] == task_id), None)

        if not rs or not task:
            continue

        # Guard: don't reassign if already busy or task already taken
        if rs["state"] != "idle":
            continue
        if task["assigned"] and task["assigned_to"] is not None:
            continue

        sx, sz = snap_to_cell_center(rs["x"], rs["z"])
        if sx != rs["x"] or sz != rs["z"]:
            set_position(robot_name, sx, sz)
            rs["x"] = sx
            rs["z"] = sz
        start = (sx, sz)
        goal  = (task["x"], task["z"])
        path  = simplify_path(astar(start, goal, rs["battery"]))

        if not path:
            path = [goal]

        rs["task_id"]       = task_id
        rs["waypoints"]     = path
        rs["battery_start"] = rs["battery"]
        rs["assign_time"]   = timestep_count
        rs["state"]         = "moving"
        rs["stuck_counter"] = 0
        rs["prev_pos"]      = None

        task["assigned"]    = True
        task["assigned_to"] = robot_name

        print(f"[Supervisor] {robot_name} → {task_id} "
              f"({len(path)} waypoints, battery={rs['battery']:.1f}%)")

    # ── 7. Debug every 200 timesteps ─────────────────────────────────────────
    if timestep_count % 200 == 0:
        print(f"\n[DEBUG] t={timestep_count}")
        for dn in ROBOT_DEFS:
            rs = robot_states[dn]
            print(f"  {dn}: state={rs['state']:18s} "
                  f"task={str(rs['task_id']):20s} "
                  f"wps={len(rs['waypoints']):2d} "
                  f"batt={rs['battery']:.1f}%")
        unassigned = [t["id"] for t in tasks
                      if not t["assigned"] and not t["done"]]
        done_count = sum(1 for t in tasks if t["done"])
        print(f"  Unassigned tasks: {unassigned}")
        print(f"  Completed so far: {metrics['tasks_completed']} / "
              f"{metrics['tasks_total']} (expired={metrics['tasks_expired']})")

    # ── 8. Summary every 1000 timesteps ──────────────────────────────────────
    if timestep_count % 1000 == 0:
        total = metrics["tasks_total"]
        done  = metrics["tasks_completed"]
        rate  = round(done / total * 100, 1) if total > 0 else 0
        print(f"\n── [{ALGORITHM_NAME}] t={timestep_count} ──")
        print(f"  Completion rate : {done}/{total} ({rate}%)")
        print(f"  Expired tasks   : {metrics['tasks_expired']}")
        print(f"  Reallocations   : {metrics['reallocation_count']}")
        print(f"  Battery failures: {metrics['robot_battery_failures']}")
        for def_name in ROBOT_DEFS:
            rs = robot_states[def_name]
            print(f"  {def_name}: {rs['battery']:.0f}% "
                  f"| {rs['state']} | task={rs['task_id']}")
        print()

    # ── 9. Export results at end ──────────────────────────────────────────────
    if timestep_count == 10000:
        export_metrics()
        print("[Supervisor] Simulation complete.")
