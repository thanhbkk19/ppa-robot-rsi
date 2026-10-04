"""GPU smoke test for the RSI loop: run this first on the host (about 5 minutes). It exits non-zero on any problem.

usage: RSI_DEVICE=cuda python scale/rsi_gpu_smoke.py

Checks, on a tiny config with every feature of D2 on:
1. the models really live on the GPU;
2. a 2-round run finishes and every logged metric is finite;
3. checkpoint / resume works on the GPU: kill after round 1, resume, finish;
4. sampling throughput, as a first speed number.
Writes nothing into scale/results; everything goes to a temporary directory.
"""
import os
import sys
import json
import math
import time
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CFG = dict(rule="argmax", K=8, distill_data="replay", n_train=100, n_eval=50, rounds=2, extra_evals=True, curriculum=True,
           her=True, demo_goals="table", demo_filter=True, train_h=[0.0, 0.3], eval_h=[0.1, 0.3], balanced=True,
           distill_lr=3e-4, distill_filter="success", bc_steps=500, distill_steps=100, critic_steps=100, seed=97)


def fail(msg):
    print("SMOKE FAIL:", msg); sys.exit(1)


def main():
    import torch
    from rsi.models import DEV, Diffusion, Critic
    from rsi.fetch import OBS_DIM, ACT_DIM
    print("torch", torch.__version__, "| device", DEV, "| cuda available", torch.cuda.is_available())
    if os.environ.get("RSI_DEVICE", "cpu") == "cuda":
        if not torch.cuda.is_available():
            fail("RSI_DEVICE=cuda but torch.cuda.is_available() is False (needs a CUDA >= 12.8 build for sm_120)")
        print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    gen, cr = Diffusion(OBS_DIM, ACT_DIM), Critic(OBS_DIM, ACT_DIM)
    if next(gen.parameters()).device.type != DEV.type or next(cr.nets[0].parameters()).device.type != DEV.type:
        fail("models are not on the requested device")

    # sampling throughput: 64 candidates x 50 envs, as in one decision of the loop
    import numpy as np
    S = np.random.randn(64 * 50, OBS_DIM).astype(np.float32)
    gen.sample(S, 0)
    t = time.time()
    for i in range(5):
        gen.sample(S, i)
    print(f"sampling 3200 candidates x 20 denoising steps: {(time.time() - t) / 5 * 1000:.0f} ms per decision")

    tmp = tempfile.mkdtemp(prefix="rsi_smoke_")
    env = dict(os.environ, RSI_RESULTS=tmp, PYTHONPATH=ROOT)
    code = "import sys, json; from rsi.loop_fetch import job; job((sys.argv[1], json.loads(sys.argv[2])))"
    from rsi.runkey import run_key
    out = os.path.join(tmp, "smoke", run_key(CFG) + ".json"); ck = out[:-5] + ".ckpt.pt"

    # (3) start, kill once the round-1 checkpoint exists, then resume
    p = subprocess.Popen([sys.executable, "-c", code, "smoke", json.dumps(CFG)], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    while p.poll() is None and not os.path.exists(ck) and time.time() - t0 < 1800:
        time.sleep(2)
    if p.poll() is not None and not os.path.exists(out):
        print(p.stdout.read()[-3000:]); fail("the run crashed before its first checkpoint (log above)")
    if p.poll() is None:
        p.kill(); p.wait(); print("killed after the round-1 checkpoint; resuming")
    r = subprocess.run([sys.executable, "-c", code, "smoke", json.dumps(CFG)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:]); fail("resume crashed (log above)")
    res = json.load(open(out))
    if not res.get("done") or [h["round"] for h in res["hist"]] != [0, 1, 2]:
        fail(f"unexpected history: {[h['round'] for h in res['hist']]}")
    for h in res["hist"]:
        for k in ("J_sys", "J_gen", "J_easy", "J_full"):
            if not math.isfinite(h[k]):
                fail(f"non-finite {k} in round {h['round']}")
    if os.path.exists(ck):
        fail("checkpoint not removed after the run finished")
    print("rounds", [(h["round"], h["J_sys"], h["J_easy"], h["J_full"]) for h in res["hist"]])
    print("SMOKE OK (temporary results in", tmp + ")")


if __name__ == "__main__":
    main()
