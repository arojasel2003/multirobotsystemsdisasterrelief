"""
greedy.py
---------
Greedy task allocation algorithm.
- For each unassigned task, finds the nearest available robot
- Simple and fast — used as the baseline to compare against CBBA

FIXES:
  - is_available battery threshold raised to 30% (was 20%) so robots
    always have enough charge to reach the charger after taking a task
  - greedy_reallocate no longer double-releases tasks (supervisor already
    releases before calling reallocate on battery/stuck triggers)
  - priority trigger correctly passes only the new task so allocation
    doesn't reassign already-running robots
"""

import math

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def distance(robot, task):
    dx = robot["x"] - task["x"]
    dz = robot["z"] - task["z"]
    return math.sqrt(dx*dx + dz*dz)


def is_available(robot):
    """Robot is available if idle and battery is comfortably above critical."""
    return robot["state"] == "idle" and robot["battery"] > 30.0

# ─── GREEDY ALLOCATOR ────────────────────────────────────────────────────────

def greedy_allocate(robots, tasks):
    """
    Greedy allocation: assign nearest available robot to each unassigned task.
    Higher priority tasks are allocated first.

    Args:
        robots: list of robot state dicts
        tasks:  list of task dicts (already filtered to unassigned+undone
                by the caller, but we guard again just in case)

    Returns:
        list of (robot_name, task_id) tuples
    """
    assignments     = []
    assigned_robots = set()

    unassigned_tasks = sorted(
        [t for t in tasks if not t["assigned"] and not t.get("done", False)],
        key=lambda t: t["priority"],
        reverse=True,
    )

    available_robots = [r for r in robots if is_available(r)]

    for task in unassigned_tasks:
        if not available_robots:
            break

        best_robot = None
        best_dist  = float("inf")

        for robot in available_robots:
            if robot["name"] in assigned_robots:
                continue
            d = distance(robot, task)
            if d < best_dist:
                best_dist  = d
                best_robot = robot

        if best_robot:
            assignments.append((best_robot["name"], task["id"]))
            assigned_robots.add(best_robot["name"])
            print(f"[Greedy] {best_robot['name']} "
                  f"(batt={best_robot['battery']:.1f}%, dist={best_dist:.2f}m)"
                  f" → {task['id']} (priority={task['priority']})")

    if not assignments:
        print("[Greedy] No assignments made.")

    return assignments


def greedy_reallocate(robots, tasks, trigger):
    """
    Called by Supervisor on a reallocation trigger.
    NOTE: for battery/stuck triggers the Supervisor has already released
    the task back to the pool before calling here — we just re-run allocation.
    For priority triggers the new task is already in the pool.

    Args:
        robots:  list of robot state dicts
        tasks:   full task list
        trigger: {"type": "battery"|"stuck"|"priority", "robot"|"task": ...}

    Returns:
        new assignments list
    """
    print(f"[Greedy] Reallocation triggered: {trigger}")
    # Run fresh greedy on current state — supervisor has already updated
    # task/robot states so we just allocate normally
    return greedy_allocate(robots, tasks)


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
        {"id": "task_0", "x": -4.0, "z": -4.0, "priority": 3, "assigned": False, "done": False},
        {"id": "task_1", "x":  0.0, "z": -4.0, "priority": 2, "assigned": False, "done": False},
        {"id": "task_2", "x":  4.0, "z": -4.0, "priority": 1, "assigned": False, "done": False},
        {"id": "task_3", "x": -4.0, "z":  4.0, "priority": 3, "assigned": False, "done": False},
        {"id": "task_4", "x":  2.0, "z":  4.0, "priority": 2, "assigned": False, "done": False},
    ]
    print("Running greedy allocation test...")
    result = greedy_allocate(test_robots, test_tasks)
    print(f"\nFinal assignments: {result}")
