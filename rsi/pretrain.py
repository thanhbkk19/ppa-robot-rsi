"""Build the cached BC generators for given seeds in parallel: python -m rsi.pretrain 1 2 3"""
import sys
from multiprocessing import Pool
from rsi.loop_fetch import pretrained, DEF


def one(s):
    c = dict(DEF); c["seed"] = s; pretrained(c); print("pretrained seed", s, flush=True)


if __name__ == "__main__":
    with Pool(4) as p:
        p.map(one, [int(x) for x in sys.argv[1:]])
