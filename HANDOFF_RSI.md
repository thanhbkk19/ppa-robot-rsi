# Handoff: RSI scale-up on the RTX 5090 host

Branch: `claude/session-hardware-info-tasocb`. Written 4 Oct 2026.

Read this first. It has:
- what to run;
- what the algorithm is and why it works;
- what has been shown so far;
- the rules that keep the results publishable.

---

## 0. TL;DR: what to do on the RTX 5090 machine

```bash
git clone <repo> && cd ppa-robot-rsi          # or: git fetch && git checkout claude/session-hardware-info-tasocb && git pull
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128   # Blackwell sm_120 needs CUDA >= 12.8
pip install -r requirements-toy.txt "gymnasium>=1.0" "gymnasium-robotics>=1.2" "mujoco>=3.1"
export MUJOCO_GL=egl RSI_DEVICE=cuda

python scale/rsi_gpu_smoke.py                  # ~5 min, must print "SMOKE OK"  (section 4.2)
mkdir -p scale/results/logs && nohup bash scale/rsi_pipeline.sh > scale/results/logs/pipeline_rsi.out 2>&1 &   # everything else, resumable
```

When it prints `RSI_PIPELINE_DONE` (estimated 5–16 h), commit and push the results (section 6):
```bash
git add scale/results/runs.csv scale/results/s0 scale/results/s1 scale/results/s2
git commit -m "S0-S2 results from the RTX 5090 host" && git push
```
Do not commit `*.pt` or `*.ckpt.pt`; they are git-ignored. After any crash, reboot or kill, rerun the same
`nohup bash scale/rsi_pipeline.sh …` command. Every step resumes where it stopped.

---

## 1. The problem

**Recursive self-improvement (RSI) for a robot policy.** The policy improves from its own experience and
selection, with no new demonstrations.

Testbed: **Fetch pick-and-place "take-off"** (MuJoCo, `rsi/fetch.py`).
- Demonstrations (600 scripted episodes, noise 0.45, filtered to the successful ones) only ever place the
  object **on the table**.
- Target goals are **10–30 cm in the air**. The demo-trained policy has **0%** success there.
- Lifting must be discovered by self-improvement while table-goal skill is kept.

Metrics: success at the final step of fixed-length episodes, on evaluation episodes no learner ever sees.

| metric | goal range |
|---|---|
| `J_target` (logged as `J_sys`) | in-air goals, 10–30 cm |
| `J_easy` | table goals, 0–5 cm |
| `J_full` | whole range, 0–30 cm |

---

## 2. The algorithm (config "D2": outcome-consistent distillation)

Code: `rsi/loop_fetch.py:run`. Learner: a diffusion policy and an MC critic ensemble (`rsi/models.py`).

```
init:  generator μ ← BC on the successful demos       (20-step DDPM, MLP 3×256, 4-step action chunks)
       critic Q    ← untrained ensemble of 2 MLPs     (Q(s,g,a) ≈ P(success), MC targets)
repeat for R rounds (10 in P20, 20 in S1):
  1. curriculum: draw 400 training goal heights; 12 bins over 0–30 cm, P(bin) ∝ m(1−m) + 0.02,
                 where m = past success rate in that bin (goals at the frontier ~50% are drawn most)
  2. act:        at each of the 13 decisions, sample 64 chunks from μ and execute argmax_a Q(s,g,a)
  3. hindsight:  every episode is also stored with goal := where the object actually ended, as a success
  4. critic:     refit Q on ALL rows (commanded + hindsight), 2000 steps
  5. DISTIL (the contribution), on the replay of all rounds so far + 10% demo rows:
       commanded-goal rows  → kept ONLY from successful episodes        (outcome filter)
       hindsight rows       → kept from ALL episodes
       weights              → equal total weight per 2.5 cm goal-height bin   (balanced)
       1500 steps at lr 3e-4                                             (larger step)
  6. evaluate on fixed held-out episodes
```

| setting | value | config key |
|---|---|---|
| candidates / selection | K = 64, argmax of the ensemble mean | `K=64, rule="argmax"` |
| distillation data | replay of all rounds, outcome-filtered | `distill_data="replay", distill_filter="success"` |
| balancing / step | equal weight per height bin; lr 3e-4 | `balanced=True, distill_lr=3e-4` |
| curriculum + hindsight | on | `curriculum=True, her=True` |
| init | filtered BC on table-only demos | `demo_goals="table", demo_filter=True` |
| per round | 400 training episodes, 200 + 100 + 100 evaluation episodes | `n_train=400, extra_evals=True` |

The full frozen configs live in `scale/rsi_specs.py`. Do not change them; see section 7.

### Why it works (the mechanism)
- Without the filter, a failed in-air episode puts the **same actions** into the distillation set twice:
  - under the commanded goal ("do this for a 20 cm goal");
  - under the hindsight goal ("do this for the table goal it reached").
- At the frontier most episodes fail, so the data tells the generator that **goal height does not matter**.
- The generator becomes goal-blind: it responds to goal height with a slope of only 0.011. Any lifting it learns
  is then applied to table goals too. This is the coupling that made every earlier frontier gain cost table-goal
  success.
- The filter makes every distilled row "these actions reached this goal". This is the GCSL data condition, and
  the outcome filter of STaR, ReST and RAFT in LLM self-training.
- With consistent data, the larger step and balancing are safe: the generator becomes goal-conditioned (slope
  0.141, 13×) and both ends improve.
- Formal note: `docs/PA_THEORY.md`, Proposition 6.

---

## 3. Evidence so far (all pre-registered, `rsi/PREREG.md`)

Control = the same loop without the filter, balancing or larger step, i.e. the best fixed-K argmax loop.
B3 = the identical loop to D2, minus the filter.

| test | seeds | comparison | J_target | J_easy | J_full |
|---|---|---|---|---|---|
| P20 held-out | 5–9 | D2 vs control | +0.226 [+0.021, +0.431] | +0.152 [+0.066, +0.238] | **+0.216 [+0.090, +0.342]** |
| S1 CPU preview | 10–19 | D2 vs control | +0.216 [+0.105, +0.327] | +0.224 [+0.110, +0.338] | **+0.223 [+0.122, +0.324]**, 10/10 seeds |
| S1 CPU preview | 10–19 | D2 vs B3 (the filter alone) | +0.141 [+0.031, +0.251] | +0.214 [+0.134, +0.294] | **+0.141 [+0.030, +0.252]** |

- Brackets are paired 95% t-intervals over seeds; the seed is the unit of replication.
- Absolute J_full on seeds 10–19: D2 0.473, B3 0.332, control 0.250.
- Mechanism (deployment-only probes on tune-seed models): generator goal slope 0.011 → 0.141. The generator
  alone, with no critic, reaches 0.47 on in-air goals; every earlier config had 0.00.
- Reports: `rsi/results/P20_REPORT.md`, `scale/results/s1cpu/REPORT.md`.

### What does not work, or is out of scope
- **Second task: push by direction (P22).** Negative. The base loop degrades there because the critic has no
  ranking signal. The claim is scoped to loops whose critic selection already moves the frontier.
- **Rejected in pre-registered tests:**
  - χ² and pessimistic selection rules;
  - goal-relative features;
  - the proxy-rewarded K bandit (FAS v1);
  - balancing or a larger step alone, on held-out seeds.
- **FAS v2** (per-goal-bin K): protects easy goals (+0.23 held-out), but the frontier effect is not confirmed. It
  is not in D2.

---

## 4. Running it on the host

### 4.1 Requirements
- Python ≥ 3.10. PyTorch ≥ 2.7 with **CUDA ≥ 12.8** (RTX 5090 = sm_120).
- `gymnasium-robotics` (FetchPickAndPlace-v4) and `mujoco`; headless rendering via `MUJOCO_GL=egl`.
- Tested here with Python 3.11, torch 2.14 (CPU), gymnasium 1.3.0, gymnasium-robotics 1.4.2, mujoco 3.2.7,
  numpy 2.4, scipy 1.17.
- Host RAM: ≈ 1.5–2 GB per S1 run and ≈ 4–5 GB per S2 run. GPU memory is small, ≈ 0.5–1 GB per run (mostly
  the CUDA context).

### 4.2 GPU smoke test (do this first)
```bash
RSI_DEVICE=cuda python scale/rsi_gpu_smoke.py
```
- It checks that the models are on the GPU, runs a tiny 2-round job, kills it after round 1, resumes it, and
  checks that every metric is finite.
- **The CUDA path has never run before**: the development container had no GPU. CPU results were verified
  bit-identical to the reference code.
- If it fails, see 4.5. The pipeline also runs it, once, and stops on failure.

### 4.3 The pipeline (`scale/rsi_pipeline.sh`)
| step | what | output |
|---|---|---|
| checks | CUDA available, unit tests, GPU smoke test | `scale/results/logs/gpu_smoke.log` |
| **S0** | profiles 1/4/8/12/16 parallel runs and picks the worker count | `scale/results/s0/profile.json` |
| **S1** | control, B3, D2 × seeds 10–29 × **20 rounds** (60 runs) | `scale/results/s1/*.json`, `verdict_tables.md` |
| **S2** | control, B3, D2 × seeds 10–19 × 10 rounds, **4× episodes, 2× width, 2× steps** (30 runs) | `scale/results/s2/…` |
| mechanism | goal-slope probe on S1 seeds 10–14 | `scale/results/s1/mechanism_slope.log` |

- Override the worker count with `WORKERS=12 bash scale/rsi_pipeline.sh`. S2 automatically uses half.
- Expected wall-clock is 5–16 h (`scale/BUDGET_RSI.md`); S0 measures the real number.

### 4.4 Monitoring
- `scale/results/runs.csv`: one row per run with status (`running(try n)`, `done`, `failed(rc …)`) and
  wall-clock.
- `scale/results/logs/s1.log` and `s2.log`: per-run output. `*_runner.log` shows the queue.
- Each result JSON is rewritten after every round (`hist` = one entry per round), so partial curves can be
  inspected at any time:
  `python scale/rsi_summarize.py scale/results/s1 --rounds 10`. This only counts finished runs.

### 4.5 Troubleshooting
| symptom | fix |
|---|---|
| `torch.cuda.is_available()` is False, or "no kernel image" | install a CUDA 12.8+ PyTorch build (sm_120) |
| MuJoCo / GL errors | `export MUJOCO_GL=egl` (or `osmesa`); pip `mujoco>=3.1` |
| a run shows `failed(rc -9)` | out of host RAM. Lower `WORKERS` and rerun the pipeline: it resumes from the per-round checkpoint |
| a device-mismatch error in the smoke test | the GPU path is untested. Check `rsi/models.py` (`DEV`, `gen_on`) and the qgrad block in `rsi/loop_fetch.py:select_fn`. `RSI_DEVICE=cpu` always works, just slower |
| need to restart one run from scratch | delete its `.json` and `.ckpt.pt` in `scale/results/<block>/` |
| GPU underused | normal: env stepping is single-core per run; parallelism comes from many runs (S0 picks the count) |

---

## 5. Pre-registered criteria (`rsi/PREREG.md`, Amendment P)

`scale/rsi_summarize.py` prints these directly; "HOLD" or "FAIL" lines appear in `verdict_tables.md`.

| id | comparison | round | criterion |
|---|---|---|---|
| S1a | D2 vs control | 10 | J_full ≥ +0.05 with paired CI > 0, J_target ≥ +0.03, J_easy ≥ −0.05 |
| S1b | D2 vs control | 20 | J_full paired CI excludes 0 (the gain persists, not just speed) |
| S1c | D2 vs B3 | 10 and 20 | J_full paired CI excludes 0 (the filter's own contribution) |
| S2a | D2 vs control | 10 (scaled) | same as S1a |

Reported without a criterion:
- control S2 vs control S1 (does 4× data close the gap?);
- the mechanism slope (prediction: D2 ≥ 5× control).

---

## 6. What to send back
Commit and push:
- `scale/results/runs.csv`;
- `scale/results/s0/profile.json`;
- `scale/results/s1/` and `scale/results/s2/`: the `*.json`, `verdict_tables.md`, `summary.json` and
  `mechanism_slope.log`;
- `scale/results/logs/gpu_smoke.log`.

Model files are not needed in git. Keep the `scale/results/s1/*.pt` locally if you want further mechanism probes
(`python -m rsi.analysis_slope`, `python -m rsi.analysis_kgoal`).

I, or the next agent session, then write `scale/results/s1/REPORT.md` and `s2/REPORT.md` with verdicts against
section 5.

---

## 7. Rules (from `CLAUDE.md`; they keep the result valid)
1. **Do not tune on seeds 10–29.**
   - Do not change any config in `scale/rsi_specs.py`, and do not pick checkpoints or rounds by test results.
   - Seeds 0–2 are the only tuning seeds.
2. **No silent changes.** Any deviation goes into `scale/DECISIONS.md` *before* running, and any new
   experiment into `rsi/PREREG.md` (amendment + prediction + criterion) before its first run.
   - Deviations include: a different seed set, fewer runs, a code change that alters results, or a new method.
3. **Report every run**, including failures and negative verdicts. Partial blocks are reported as partial.
4. **Budgets are matched.** All methods get the same episodes, rounds and network sizes inside a block.
5. Do not commit datasets, checkpoints or videos.

---

## 8. After the results: decision tree
- **S1a + S1b + S1c hold, and S2a holds** → the core claim is solid. Next, in this order:
  - (a) a second task where critic selection works, e.g. pick-and-place to far/high goals or another
    goal-conditioned manipulation task (OGBench cube, Franka kitchen goals);
  - (b) D2 + per-goal K (P21: the K-inversion still exists in D2 models, where K = 4 beats K = 64 at deployment);
  - (c) paper write-up: diagnosis (Proposition 6) + method + mechanism probes.
- **S1a holds but S1b fails** (the control catches up by round 20) → the method is a speed-up, not a better
  asymptote. Report it that way and test longer horizons.
- **S2a fails** (scale removes the gap) → report it. The label contradiction may matter only at small data /
  capacity.
- **Anything crashes or looks implausible** (e.g. GPU results far from the CPU preview on seeds 10–19 at round
  10) → check the smoke log and compare a GPU run against the CPU preview JSON for the same seed
  (`scale/results/s1cpu/`) before trusting the numbers.

---

## 9. File map
| path | what |
|---|---|
| `rsi/loop_fetch.py` | the loop (`run`), all config keys (`DEF`), checkpoint/resume, `job` |
| `rsi/models.py` | diffusion generator, critic ensemble, `RSI_DEVICE` switch, `load_system` |
| `rsi/fetch.py` | Fetch envs, goal placement (pnp height / push angle), scripted demos |
| `rsi/runkey.py` | result file names (long configs → hashed key) |
| `scale/rsi_specs.py` → `scale/specs/rsi_s1.jsonl`, `rsi_s2.jsonl` | the frozen scale-up grid |
| `scale/rsi_run.py` | parallel runner + registry `scale/results/runs.csv` |
| `scale/rsi_profile.py` | S0 profiler |
| `scale/rsi_gpu_smoke.py` | GPU smoke test |
| `scale/rsi_summarize.py` | paired verdict tables |
| `scale/rsi_pipeline.sh` | runs everything, resumable |
| `scale/RSI_SCALEUP.md`, `scale/BUDGET_RSI.md`, `scale/DECISIONS.md` (D9) | runbook, budget, decisions |
| `rsi/PREREG.md` | every pre-registration and verdict (Amendments A–R) |
| `rsi/results/P20_REPORT.md`, `scale/results/s1cpu/REPORT.md` | the positive results |
| `docs/PA_THEORY.md`, `docs/RSI_STATUS.md` | theory notes, overall status |
| `rsi/analysis_*.py` | mechanism probes (goal slope M8, K-by-goal M6, push M9, …) |

## 10. Note for a coding agent on the host
- Follow `CLAUDE.md` and section 7.
- Your job is to run the pipeline, fix *infrastructure* problems (install, CUDA, memory) and report.
  - An infrastructure fix that could change results, e.g. a code change in `rsi/`, must first be checked on CPU
    against the reference: `scale/results/s1cpu/` JSONs for the same seed and config.
  - Log it in `scale/DECISIONS.md`.
- Do not change the algorithm or its configs to "improve" the numbers.
