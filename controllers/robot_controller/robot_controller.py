"""
robot_controller.py
-------------------
Lightweight robot controller for the disaster relief simulation.
Movement is handled entirely by the Supervisor (translation setting).
This controller tracks battery and relays state via customData.

FIXES:
  - BATTERY_CRITICAL raised to 30% to match supervisor/allocator thresholds
  - Battery drain rates tuned: moving=0.008/step, moving_to_charge=0.004/step
    (was 0.005/0.0025 — too slow, robots never hit low-battery trigger)
  - Removed stale cmd parsing: supervisor manages state via customData JSON;
    robot only reads "cmd" if present, ignores malformed or empty data cleanly
  - State machine cleaned up: charging->idle transition only triggers when
    battery is truly full (>= 99.9) not just >= 100 (float precision fix)
"""

from controller import Robot
import json

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

BATTERY_MAX      = 100.0
BATTERY_CRITICAL = 30.0     # must match supervisor BATTERY_CRITICAL
BATTERY_DRAIN    = 0.008    # % per timestep while moving to task
BATTERY_DRAIN_CHARGE = 0.004  # % per timestep while moving to charger
BATTERY_CHARGE   = 1.0      # % per timestep while at charging station

# ─── SETUP ───────────────────────────────────────────────────────────────────

robot    = Robot()
timestep = int(robot.getBasicTimeStep())
name     = robot.getName()

# ─── STATE ───────────────────────────────────────────────────────────────────

battery = BATTERY_MAX
state   = "idle"   # idle | moving | moving_to_charge | charging

print(f"[{name}] Started. Battery: {battery:.1f}%")

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def report_status(bat, current_state, task_done=False):
    status = {
        "battery":   round(bat, 2),
        "state":     current_state,
        "task_done": task_done,
    }
    robot.setCustomData(json.dumps(status))


def read_command():
    raw = robot.getCustomData()
    if not raw or raw.strip() == "":
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "cmd" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

while robot.step(timestep) != -1:

    task_done = False

    # ── Read command from Supervisor ──────────────────────────────────────────
    cmd = read_command()
    if cmd:
        c = cmd.get("cmd", "")
        if c == "go" and state not in ("charging", "moving_to_charge"):
            state = "moving"
        elif c == "charge":
            state = "moving_to_charge"
        elif c == "idle":
            state = "idle"
        elif c == "arrived":
            task_done = True
            state     = "idle"
        elif c == "at_charger":
            state = "charging"

    # ── Battery logic ─────────────────────────────────────────────────────────
    if state == "moving":
        battery = max(0.0, battery - BATTERY_DRAIN)

    elif state == "moving_to_charge":
        battery = max(0.0, battery - BATTERY_DRAIN_CHARGE)

    elif state == "charging":
        battery = min(BATTERY_MAX, battery + BATTERY_CHARGE)
        if battery >= 99.9:
            battery = BATTERY_MAX
            print(f"[{name}] Fully charged!")
            state = "idle"

    # ── Battery critical override ─────────────────────────────────────────────
    if (battery <= BATTERY_CRITICAL
            and state not in ("charging", "moving_to_charge")):
        print(f"[{name}] Battery critical ({battery:.1f}%)! Heading to charger.")
        state = "moving_to_charge"

    # ── Report back to Supervisor ─────────────────────────────────────────────
    report_status(battery, state, task_done)
