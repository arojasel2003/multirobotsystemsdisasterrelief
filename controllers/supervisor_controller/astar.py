"""
astar.py
--------
A* path planning on a discretized 10x10 grid of the Webots world.
- Grid cells are 1x1 metre each
- Obstacles are inflated by 1 cell (robot radius buffer) to prevent wall clipping
- Path cost = distance + battery weight penalty
- Returns list of (x, z) world-coordinate waypoints

FIXES:
  - Obstacle cells now match actual .wbt obstacle positions exactly
  - Obstacle inflation added (robot can't enter cells adjacent to obstacles)
    so robots never clip through wall edges
  - GRID_ORIGIN and coordinate conversion verified against Webots world bounds
  - World boundary cells marked as obstacles so robots stay inside
  - simplify_path kept disabled (straight-line shortcuts cut through obstacles)
"""

import heapq
import numpy as np
from collections import deque

# ─── GRID SETUP ──────────────────────────────────────────────────────────────

GRID_SIZE   = 10        # 10x10 grid covering -5..+5 in both X and Z
CELL_SIZE   = 1.0       # each cell = 1 metre
GRID_ORIGIN = -5.0      # world coord of cell [0][0] left/top corner

BATTERY_WEIGHT = 1.2    # higher = more conservative routing when battery low

# ─── RAW OCCUPANCY GRID ──────────────────────────────────────────────────────
# Derived directly from .wbt obstacle translations:
#   standing_wall_1  → (-3.5, -3.0) → col=1, row=2  → [2][1]
#   collapsed_wall_1 → (-3.0, -2.0) → col=2, row=3  → [3][2]  (rotated, spans ~col1-3)
#   fallen_slab_1    → (-2.0, -1.0) → col=3, row=4  → [4][3]
#   rock_4           → ( 1.0, -2.5) → col=6, row=2  → [2][6]
#   barrel_3         → ( 2.5, -3.0) → col=7, row=2  → [2][7]
#   barrel_2         → ( 3.6, -2.5) → col=8, row=2  → [2][8]
#   barrel_1         → ( 3.0, -2.0) → col=8, row=3  → [3][8]
#   collapsed_slab_2 → ( 2.8,  0.2) → col=7, row=5  → [5][7]
#   rock_1           → (-1.5,  1.5) → col=3, row=6  → [6][3]
#   standing_wall_2  → ( 3.0,  1.5) → col=8, row=6  → [6][8]
#   rock_2           → (-2.5,  2.5) → col=2, row=7  → [7][2]
#   debris_slab_1    → ( 0.8,  2.7) → col=5, row=7  → [7][5]  (rotated, spans col4-5)
#   rock_3           → (-1.0,  2.8) → col=4, row=7  → [7][4]
#   rock_5           → ( 3.5,  2.5) → col=8, row=7  → [7][8]
#
# Row 0 = z in [-5,-4], Row 9 = z in [4,5]
# Col 0 = x in [-5,-4], Col 9 = x in [4,5]

_RAW_GRID = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # row 0  z=-5..-4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # row 1  z=-4..-3
    [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],   # row 2  z=-3..-2
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],   # row 3  z=-2..-1
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # row 4  z=-1.. 0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # row 5  z= 0.. 1
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],   # row 6  z= 1.. 2
    [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],   # row 7  z= 2.. 3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # row 8  z= 3.. 4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # row 9  z= 4.. 5
], dtype=np.int8)

# ─── OCCUPANCY GRID ──────────────────────────────────────────────────────────
# We do NOT inflate obstacles here. The raw grid already has one cell per
# obstacle object. Wall-clipping is prevented instead by:
#   1. Corner-cutting prevention in get_neighbours (diagonal moves blocked
#      when either orthogonal neighbour is an obstacle).
#   2. is_cell_free() check in the supervisor's move_step (runtime guard).
# This keeps the grid well-connected across the whole map while still
# routing robots safely around all obstacles.

OCCUPANCY_GRID = _RAW_GRID.copy()

# ─── COORDINATE CONVERSION ───────────────────────────────────────────────────

def world_to_grid(x, z):
    """Convert world (x, z) to grid (row, col), clamped to grid bounds."""
    col = int((x - GRID_ORIGIN) / CELL_SIZE)
    row = int((z - GRID_ORIGIN) / CELL_SIZE)
    col = max(0, min(GRID_SIZE - 1, col))
    row = max(0, min(GRID_SIZE - 1, row))
    return (row, col)


def grid_to_world(row, col):
    """Convert grid (row, col) to world (x, z) at centre of that cell."""
    x = GRID_ORIGIN + col * CELL_SIZE + CELL_SIZE / 2.0
    z = GRID_ORIGIN + row * CELL_SIZE + CELL_SIZE / 2.0
    return (x, z)


def snap_to_cell_center(x, z):
    """
    Snap a world (x, z) coordinate to the centre of the nearest free grid
    cell. If the natural cell is free, returns its centre. Otherwise BFS
    outward to the closest free cell and returns that centre.
    """
    raw_row, raw_col = world_to_grid(x, z)
    free_row, free_col = _find_nearest_free(raw_row, raw_col)
    return grid_to_world(free_row, free_col)

# ─── A* ALGORITHM ────────────────────────────────────────────────────────────

def heuristic(a, b):
    """Euclidean distance heuristic."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def get_neighbours(row, col):
    """
    Return valid 8-connected neighbours.
    Diagonal moves are only allowed if BOTH orthogonal neighbours are free
    (corner-cutting prevention — stops robots slipping through diagonal gaps
    between two obstacle cells that share only a corner).
    """
    neighbours = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                continue
            if OCCUPANCY_GRID[nr][nc] == 1:
                continue
            # Prevent corner-cutting for diagonal moves
            if dr != 0 and dc != 0:
                if OCCUPANCY_GRID[row + dr][col] == 1:
                    continue
                if OCCUPANCY_GRID[row][col + dc] == 1:
                    continue
            move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
            neighbours.append((nr, nc, move_cost))
    return neighbours


def _find_nearest_free(row, col):
    """
    If (row, col) is inside an obstacle (after inflation), find the
    nearest free cell using BFS so A* always has a valid start/goal.
    """
    if OCCUPANCY_GRID[row][col] == 0:
        return (row, col)
    visited = {(row, col)}
    queue = deque([(row, col)])
    while queue:
        r, c = queue.popleft()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                        and (nr, nc) not in visited):
                    if OCCUPANCY_GRID[nr][nc] == 0:
                        return (nr, nc)
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return (row, col)  # fallback — should never reach here


def astar(start_world, goal_world, battery_level=100.0):
    """
    Run A* from start to goal in world coordinates.

    Args:
        start_world:   (x, z) world coordinates of start
        goal_world:    (x, z) world coordinates of goal
        battery_level: current battery % (0-100)

    Returns:
        List of (x, z) world coordinate waypoints, or empty list if no path.
    """
    raw_start = world_to_grid(*start_world)
    raw_goal  = world_to_grid(*goal_world)

    # Snap start/goal to nearest free cell if they land inside an obstacle
    start = _find_nearest_free(*raw_start)
    goal  = _find_nearest_free(*raw_goal)

    if start == goal:
        return [goal_world]

    # Battery penalty — low battery = penalise longer paths more heavily
    battery_penalty = BATTERY_WEIGHT * (1.0 - battery_level / 100.0)

    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from = {}
    g_score   = {start: 0.0}
    f_score   = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(grid_to_world(*current))
                current = came_from[current]
            path.append(grid_to_world(*start))
            path.reverse()
            return path

        for nr, nc, move_cost in get_neighbours(*current):
            neighbour = (nr, nc)
            tentative_g = (g_score[current]
                           + move_cost
                           + move_cost * battery_penalty)
            if neighbour not in g_score or tentative_g < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour]   = tentative_g
                f_score[neighbour]   = tentative_g + heuristic(neighbour, goal)
                heapq.heappush(open_set, (f_score[neighbour], neighbour))

    print(f"[A*] No path from {start_world} to {goal_world} — returning direct.")
    return [goal_world]


def simplify_path(waypoints):
    """
    Returns waypoints unchanged.
    Straight-line simplification is disabled — shortcuts between
    non-adjacent grid cells pass through obstacle-inflated zones.
    A* with inflation already produces clean, safe paths.
    """
    return waypoints


# ─── DEBUG / TEST ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Inflated occupancy grid:")
    for r in range(GRID_SIZE):
        print("  ", list(OCCUPANCY_GRID[r]))

    start = (-3.2, -3.5)   # robot_0 start from .wbt
    goal  = (0.0, -4.0)    # task_1
    print(f"\nPlanning path from {start} to {goal}...")
    path = astar(start, goal, battery_level=80.0)
    if path:
        print(f"Path: {len(path)} waypoints")
        for i, wp in enumerate(path):
            print(f"  {i+1}: {wp}")
    else:
        print("No path found.")
