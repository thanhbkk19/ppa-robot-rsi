"""File key of a run config (shared by rsi/loop_fetch.py and scale/rsi_run.py; no heavy imports)."""
import hashlib


def run_key(cfg):
    """Readable key 'k1v1_k2v2...'. Keys too long for a file name (> 200 chars) become a stable hash + seed;
    the full config is stored inside the result JSON either way."""
    key = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
    if len(key) > 200:
        key = "cfg" + hashlib.sha1(key.encode()).hexdigest()[:12] + f"_seed{cfg.get('seed')}"
    return key
