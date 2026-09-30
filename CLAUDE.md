# Instructions for the coding agent (read fully before touching code)

You are scaling up **Prediction-Powered Amplification (PPA)** from a CPU toy to simulated robot manipulation on one host with **1× RTX 5090 (32 GB)**. The research spec is `docs/PPA_SPEC.md`, and it is authoritative. Toy evidence is in `docs/RESULTS_TOY.md`; the code is in `toy/`.

The goal is **not** to make PPA win. It is to find out, with fair baselines, whether PPA beats *plug-in reward-model fine-tuning* when a self-verifier is gamed. A clean negative result is a valid outcome.

## Non-negotiable rules
1. **Label contract** (`ppa/labels.py`). Use `ppa_labels` / `label_batch`; do not re-implement them.
   - Never clip or normalize the DR labels Ỹ.
   - Never refit r̂ on a batch before labelling that batch.
   - The anchor decision may use only pre-label information.
   - p > 0 always.
   - Run `pytest -q` after any change to `ppa/`.
2. **Ground truth is read only through the anchor oracle** for anchored episodes. Evaluation success (reporting only) is computed on separate evaluation episodes that no learner ever sees.
3. **Seeds.** Tune every method on seeds 0–2 with the same tuning budget per method (the same number of configs). Report on held-out seeds 3–7 or more. Never select checkpoints or configs using held-out seeds. The unit of replication is a seed, not an episode.
4. **Budgets are matched** across methods: the same environment episodes, the same anchor budget (for methods that use anchors), and the same generator and checkpoint. Log wall-clock too.
5. **Baselines first.** A milestone that changes the method is blocked until the unmodified backbone reproduces its published numbers within tolerance (document the tolerance before running).
6. **No silent changes.** If you must deviate from `docs/PPA_SPEC.md` (framework, task, learner), write the reason in `scale/DECISIONS.md` before running.
7. **Commit discipline.** Small commits; results as JSON under `scale/results/<milestone>/`, plus a short `REPORT.md` per milestone with tables and a plain verdict against the pre-registered criterion. Do not commit datasets, checkpoints or videos (see `.gitignore`).

## Milestones (do them in order; each has a stop condition)
- **M0: environment.**
  - Verify the GPU (`nvidia-smi`; PyTorch ≥ 2.7 with CUDA ≥ 12.8 is required for Blackwell sm_120; JAX needs a recent CUDA-12 jaxlib).
  - `pip install -r requirements-toy.txt && pytest -q && cd toy && python ms2.py` must import cleanly.
  - Rerun one toy config (`python -c "from ms2 import run; print(run('critic', label='dr', hack=True, lam=0.1, seed=3))"`) and check it lands near the table in the spec.
  - Headless MuJoCo: `MUJOCO_GL=egl`.
- **M1: backbone reproduction.**
  - Clone PA-RL (https://github.com/MaxSobolMark/PolicyAgnosticRL) into `third_party/` (git submodule), and check whether it is JAX or PyTorch.
  - Get pretrained diffusion checkpoints for Robomimic (DPPO release: https://diffusion-ppo.github.io/).
  - Reproduce PA-RL online fine-tuning with the **true reward** on one task (Square first).
  - Profile episodes/hour and GPU memory. Write `scale/BUDGET.md`: episodes per round, rounds, seeds, and total wall-clock for the M4 grid. If the grid exceeds about 10 GPU-days, shrink tasks or seeds and document it.
  - **Stop** if the reproduction fails after reasonable effort, and report.
- **M2: gameable verifiers.** Implement a self-verifier V that replaces the reward in the backbone:
  - (a) a success classifier trained on demos plus a few failures
  - (c) injected false positives on a specified state region, which must be reachable by the pretrained policy
  - (b) a VLM judge is optional, later
  - Show that PA-RL with V only is **hacked**: true success drops, or the gap J_self − J grows. **Stop** if no regime is hackable; the benchmark would then not test the claim.
- **M3: PPA in the backbone.**
  - Feed Ỹ from `label_batch` wherever the backbone consumes the terminal reward.
  - Implement two critic variants: MC-return targets (C1 holds) and the native TD critic (C1 only approximate). Report both.
  - Unit-test that, with anchor budget p = 1, PPA reduces exactly to the true-reward backbone.
- **M4: comparison grid** (tasks × verifier regimes × anchor budgets {1, 2, 5}% × methods × seeds), following `scale/configs/grid.yaml`.
  - Methods: oracle, V-only, anchors-only, naive mix, **plug-in reward-model fine-tuning on anchors**, PPA-DR (uniform and claimed-success anchoring; r̂ = 0 and learned r̂), filtered BC, DSRL (true reward, if time allows), frozen-generator selector.
  - Pre-registered verdict:
    - **GO:** PPA at ≤ 2% anchors is within 0.05 of oracle final success AND beats plug-in fine-tuning by ≥ 0.10 in at least one regime where the verifier's failure is invisible to the reward-model features.
    - **NO-GO:** plug-in matches PPA everywhere.
- **M5 (only after GO): gate and theory experiments.**
  - Implement the anytime-valid PPI acceptance gate (spec §2).
  - Test the learnability threshold n_eff·K·m ≳ c by sweeping K and the anchor budget.

## Hardware notes (1× RTX 5090)
- Run one training job per GPU at a time, plus CPU-parallel env workers. Do not oversubscribe VRAM with parallel seeds unless profiling shows headroom.
- Every job must be resumable from checkpoints (seeds run for hours). Write results incrementally.
- Keep a run registry: `scale/results/runs.csv` with run id, git SHA, config hash, seed, status, and wall-clock.

## What you must not do
- Do not claim novelty or superiority in reports. Report numbers against the pre-registered criteria only.
- Do not reduce demonstrations to manufacture headroom. Use checkpoints with non-saturated success.
- Do not tune PPA more than the baselines.
