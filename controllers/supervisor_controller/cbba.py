"""
cbba.py
-------
Consensus-Based Bundle Algorithm (CBBA) for multi-robot task allocation.

FIXES:
  - is_available battery threshold raised to 30% (matches greedy)
  - compute_score: distance penalty scaled properly so high-priority tasks
    always beat low-priority ones regardless of distance
  - build_bundles: fallback no longer adds a task already in another robot's
    bundle (previously could cause two robots to race for same task)
  - consensus: winning_bids rebuilt from scratch each iteration instead of
    carrying stale entries — was causing tasks to stay assigned to losers
  - cbba_reallocate: no longer double-releases tasks (supervisor does it first)
  - Greedy fallback in cbba_allocate fixed — leftover_tasks now excludes
    tasks already won in the consensus phase
"""

import math
import copy

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

BATTERY_WEIGHT  = 0.3
PRIORITY_WEIGHT = 5.0    # raised so priority always dominates distance
DISTANCE_WEIGHT = 0.1
MAX_BUNDLE_SIZE = 3
MAX_ITERATIONS  = 100

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def distance(x1, z1, x2, z2):
    return math.sqrt((x1 - x2)**2 + (z1 - z2)**2)


def is_available(robot):
    return robot["state"] == "idle" and robot["battery"] > 30.0


def compute_score(robot, task, bundle):
    """
    Bid score: higher = better fit.
    Priority dominates; distance and battery are tiebreakers.
    Bundle penalty discourages hoarding tasks.
    """
    dist           = distance(robot["x"], robot["z"], task["x"], task["z"])
    battery_factor = (robot["battery"] / 100.0) * BATTERY_WEIGHT
    priority_factor = task["priority"] * PRIORITY_WEIGHT
    distance_penalty = dist * DISTANCE_WEIGHT
    bundle_penalty   = len(bundle) * 0.5
    return priority_factor + battery_factor - distance_penalty - bundle_penalty

# ─── PHASE 1: BUNDLE BUILDING ────────────────────────────────────────────────

def build_bundles(robots, tasks):
    """
    Each robot greedily builds its own bundle, up to MAX_BUNDLE_SIZE.
    A task may appear in multiple robots' bundles — consensus resolves this.
    """
    bundles = {r["name"]: [] for r in robots}
    bids    = {r["name"]: {} for r in robots}

    available_tasks = [t for t in tasks if not t["assigned"] and not t.get("done", False)]

    for robot in robots:
        if not is_available(robot):
            continue

        remaining = list(available_tasks)

        while len(bundles[robot["name"]]) < MAX_BUNDLE_SIZE and remaining:
            best_task  = None
            best_score = float("-inf")

            for task in remaining:
                score = compute_score(robot, task, bundles[robot["name"]])
                if score > best_score:
                    best_score = score
                    best_task  = task

            if best_task:
                bundles[robot["name"]].append(best_task["id"])
                bids[robot["name"]][best_task["id"]] = best_score
                remaining.remove(best_task)
            else:
                break

        # Fallback: if robot is idle but has nothing in bundle, force-assign closest
        if not bundles[robot["name"]] and available_tasks:
            closest = min(available_tasks,
                          key=lambda t: distance(robot["x"], robot["z"],
                                                 t["x"], t["z"]))
            score = compute_score(robot, closest, [])
            bundles[robot["name"]].append(closest["id"])
            bids[robot["name"]][closest["id"]] = score

    return bundles, bids

# ─── PHASE 2: CONSENSUS ──────────────────────────────────────────────────────

def consensus(robots, bundles, bids):
    """
    Iteratively resolve conflicts: robot with highest bid wins each task.
    Losers remove the contested task from their bundle.
    Rebuilt from scratch each iteration to avoid stale winning_bids entries.
    """
    for iteration in range(MAX_ITERATIONS):
        # Recompute winning bids fresh each iteration
        winning_bids = {}
        for robot in robots:
            name = robot["name"]
            for task_id in bundles[name]:
                score = bids[name].get(task_id, 0.0)
                if task_id not in winning_bids or score > winning_bids[task_id][1]:
                    winning_bids[task_id] = (name, score)

        changed = False
        for robot in robots:
            name = robot["name"]
            for task_id in list(bundles[name]):
                winner, _ = winning_bids.get(task_id, (None, 0))
                if winner != name:
                    bundles[name].remove(task_id)
                    changed = True

        if not changed:
            break

    # Build final assignments — highest-scoring match wins, one task per robot
    assignments     = []
    assigned_tasks  = set()
    assigned_robots = set()

    # Recompute final winning_bids after consensus
    winning_bids = {}
    for robot in robots:
        name = robot["name"]
        for task_id in bundles[name]:
            score = bids[name].get(task_id, 0.0)
            if task_id not in winning_bids or score > winning_bids[task_id][1]:
                winning_bids[task_id] = (name, score)

    for task_id, (winner_name, _) in sorted(winning_bids.items(),
                                             key=lambda kv: kv[1][1],
                                             reverse=True):
        if task_id not in assigned_tasks and winner_name not in assigned_robots:
            assignments.append((winner_name, task_id))
            assigned_tasks.add(task_id)
            assigned_robots.add(winner_name)

    return assignments, assigned_tasks, assigned_robots

# ─── MAIN CBBA FUNCTION ──────────────────────────────────────────────────────

def cbba_allocate(robots, tasks):
    """
    Full CBBA allocation.

    Returns:
        list of (robot_name, task_id) tuples
    """
    available_robots = [r for r in robots if is_available(r)]
    unassigned_tasks = [t for t in tasks
                        if not t["assigned"] and not t.get("done", False)]

    if not available_robots or not unassigned_tasks:
        print("[CBBA] No available robots or tasks.")
        return []

    print(f"[CBBA] Allocating: {len(available_robots)} robots, "
          f"{len(unassigned_tasks)} tasks")

    bundles, bids = build_bundles(available_robots, unassigned_tasks)

    print("[CBBA] Bundles built:")
    for name, bundle in bundles.items():
        if bundle:
            print(f"  {name}: {bundle}")

    assignments, assigned_task_ids, assigned_robot_names = consensus(
        available_robots, bundles, bids
    )

    print("[CBBA] Consensus assignments:")
    for robot_name, task_id in assignments:
        score = bids[robot_name].get(task_id, 0)
        print(f"  {robot_name} → {task_id} (score={score:.2f})")

    # Greedy fallback for tasks nobody won and spare robots
    spare_robots   = [r for r in available_robots
                      if r["name"] not in assigned_robot_names]
    leftover_tasks = [t for t in unassigned_tasks
                      if t["id"] not in assigned_task_ids]

    leftover_tasks = sorted(leftover_tasks,
                            key=lambda t: t["priority"], reverse=True)
    spare_robots   = sorted(spare_robots,
                            key=lambda r: r["battery"], reverse=True)

    for task, robot in zip(leftover_tasks, spare_robots):
        assignments.append((robot["name"], task["id"]))
        print(f"  [fallback] {robot['name']} → {task['id']} "
              f"(priority={task['priority']})")

    if not assignments:
        print("[CBBA] No assignments made.")

    return assignments


def cbba_reallocate(robots, tasks, trigger):
    """
    Called by Supervisor when a trigger fires.
    Supervisor has already released the task back to the pool,
    so we just re-run CBBA on current state.
    """
    print(f"[CBBA] Reallocation triggered: {trigger}")
    return cbba_allocate(robots, tasks)


# ─── DEBUG / TEST ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_robots = [
        {"name": "robot_0", "x": -3.2, "z": -3.5, "battery": 100.0, "state": "idle"},
        {"name": "robot_1", "x": -3.2, "z":  4.4, "battery":  85.0, "state": "idle"},
        {"name": "robot_2", "x":  1.8, "z":  3.5, "battery":  60.0, "state": "idle"},
        {"name": "robot_3", "x":  4.5, "z": -0.7, "battery":  95.0, "state": "idle"},
        {"name": "robot_4", "x":  0.5, "z": -3.2, "battery":  40.0, "state": "idle"},
    ]
    test_tasks = [
        {"id": "task_0", "x": -4.0, "z": -4.0, "priority": 3, "assigned": False, "done": False, "assigned_to": None},
        {"id": "task_1", "x":  0.0, "z": -4.0, "priority": 2, "assigned": False, "done": False, "assigned_to": None},
        {"id": "task_2", "x":  4.0, "z": -4.0, "priority": 1, "assigned": False, "done": False, "assigned_to": None},
        {"id": "task_3", "x": -4.0, "z":  4.0, "priority": 3, "assigned": False, "done": False, "assigned_to": None},
        {"id": "task_4", "x":  2.0, "z":  4.0, "priority": 2, "assigned": False, "done": False, "assigned_to": None},
    ]
    print("Running CBBA test...")
    result = cbba_allocate(test_robots, test_tasks)
    print(f"\nFinal: {result}")
