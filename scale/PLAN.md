# Scale-up plan
Follow the milestones M0–M5 in `../CLAUDE.md`. This folder holds:
- `configs/grid.yaml`: the M4 grid (budgets are filled in after M1 profiling)
- `BUDGET.md`: measured throughput and the planned GPU-days (written in M1)
- `DECISIONS.md`: every deviation from docs/PPA_SPEC.md, with its reason (written before running)
- `results/<milestone>/`: JSON results plus a REPORT.md with the verdict against pre-registered criteria
- `results/runs.csv`: run registry (run id, git SHA, config hash, seed, status, wall-clock)
