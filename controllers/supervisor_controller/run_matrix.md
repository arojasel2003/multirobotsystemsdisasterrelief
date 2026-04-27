# Run Matrix — 18 Webots Runs

Edit the four flags at the top of `supervisor_controller.py` before each run, then
start Webots, wait for `[Supervisor] Simulation complete.`, and confirm the JSON
was written. Check off each row as you go.

## Flag reference

```
USE_CBBA   = False | True
STRESS_MODE = False | True
SCALE_MODE  = False | True
SEED        = 1 | 2 | 3
```

`SCALE_MODE=True` overrides the robot count to 8.  
`STRESS_MODE=True` enables high-density / rapid-expiry task generation.  
Both can be False simultaneously (baseline).

---

## Checklist

| # | algo   | mode     | seed | USE_CBBA | STRESS_MODE | SCALE_MODE | output file                              | done |
|---|--------|----------|------|----------|-------------|------------|------------------------------------------|------|
| 1 | greedy | baseline | 1    | False    | False       | False      | results_greedy_baseline_seed1.json       | [ ]  |
| 2 | greedy | baseline | 2    | False    | False       | False      | results_greedy_baseline_seed2.json       | [ ]  |
| 3 | greedy | baseline | 3    | False    | False       | False      | results_greedy_baseline_seed3.json       | [ ]  |
| 4 | cbba   | baseline | 1    | True     | False       | False      | results_cbba_baseline_seed1.json         | [ ]  |
| 5 | cbba   | baseline | 2    | True     | False       | False      | results_cbba_baseline_seed2.json         | [ ]  |
| 6 | cbba   | baseline | 3    | True     | False       | False      | results_cbba_baseline_seed3.json         | [ ]  |
| 7 | greedy | stress   | 1    | False    | True        | False      | results_greedy_stress_seed1.json         | [ ]  |
| 8 | greedy | stress   | 2    | False    | True        | False      | results_greedy_stress_seed2.json         | [ ]  |
| 9 | greedy | stress   | 3    | False    | True        | False      | results_greedy_stress_seed3.json         | [ ]  |
|10 | cbba   | stress   | 1    | True     | True        | False      | results_cbba_stress_seed1.json           | [ ]  |
|11 | cbba   | stress   | 2    | True     | True        | False      | results_cbba_stress_seed2.json           | [ ]  |
|12 | cbba   | stress   | 3    | True     | True        | False      | results_cbba_stress_seed3.json           | [ ]  |
|13 | greedy | scale8   | 1    | False    | False       | True       | results_greedy_scale8_seed1.json         | [ ]  |
|14 | greedy | scale8   | 2    | False    | False       | True       | results_greedy_scale8_seed2.json         | [ ]  |
|15 | greedy | scale8   | 3    | False    | False       | True       | results_greedy_scale8_seed3.json         | [ ]  |
|16 | cbba   | scale8   | 1    | True     | False       | True       | results_cbba_scale8_seed1.json           | [ ]  |
|17 | cbba   | scale8   | 2    | True     | False       | True       | results_cbba_scale8_seed2.json           | [ ]  |
|18 | cbba   | scale8   | 3    | True     | False       | True       | results_cbba_scale8_seed3.json           | [ ]  |

---

## After all runs

```bash
cd controllers/supervisor_controller
python3 results.py
```

Produces:
- `comparison_baseline.png`
- `comparison_stress.png`
- `comparison_scale8.png`
- Printed summary table with mean ± std for every metric, both algos side-by-side.
