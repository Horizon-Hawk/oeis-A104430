"""Independent cross-check for A104430-style counts, with the user's mod-4 bucket split.

Different algorithm from ap_par.py:
  * Branch on whichever END is tighter: the lowest free number (must START a group) or the
    highest free number (must END a group) -- pick the one with fewer options. Every solution
    has exactly one group through that number, so each solution is counted once.
  * Every solution is tallied by its fingerprint: k0 = groups with jump = 0 mod 4,
    k2 = groups with jump = 2 mod 4 (the rest have odd jumps). The bucket totals must sum to a(n).
  * Work is cut into batches; each finished batch is appended to a checkpoint file, so a crash
    or stop loses at most one batch, and a rerun resumes.
Usage: python ap_buckets.py n_from n_to [depth] [known,comma,list] [batch]
"""
import json
import os
import sys
import time
import numpy as np
from numba import njit, prange

T = 4


@njit(cache=True)
def n_start(used, lo, hi):
    c = 0
    d = 1
    while lo + 3 * d <= hi:
        if used[lo + d] == 0 and used[lo + 2 * d] == 0 and used[lo + 3 * d] == 0:
            c += 1
        d += 1
    return c


@njit(cache=True)
def n_end(used, lo, hi):
    c = 0
    d = 1
    while hi - 3 * d >= lo:
        if used[hi - d] == 0 and used[hi - 2 * d] == 0 and used[hi - 3 * d] == 0:
            c += 1
        d += 1
    return c


@njit(cache=True)
def rec(used, left, lo, hi, k0, k2, hist):
    if left == 0:
        hist[k0, k2] += 1
        return
    while used[lo]:
        lo += 1
    while used[hi]:
        hi -= 1
    a = n_start(used, lo, hi)
    if a == 0:
        return
    b = n_end(used, lo, hi)
    if b == 0:
        return
    d = 1
    if a <= b:                                   # groups starting at lo
        while lo + 3 * d <= hi:
            if used[lo + d] == 0 and used[lo + 2 * d] == 0 and used[lo + 3 * d] == 0:
                for j in range(4):
                    used[lo + j * d] = 1
                r = d % 4
                rec(used, left - 1, lo, hi, k0 + (r == 0), k2 + (r == 2), hist)
                for j in range(4):
                    used[lo + j * d] = 0
            d += 1
    else:                                        # groups ending at hi
        while hi - 3 * d >= lo:
            if used[hi - d] == 0 and used[hi - 2 * d] == 0 and used[hi - 3 * d] == 0:
                for j in range(4):
                    used[hi - j * d] = 1
                r = d % 4
                rec(used, left - 1, lo, hi, k0 + (r == 0), k2 + (r == 2), hist)
                for j in range(4):
                    used[hi - j * d] = 0
            d += 1


@njit(parallel=True, cache=True)
def run_batch(states, meta, n):
    out = np.zeros((states.shape[0], n + 1, n + 1), np.int64)
    for i in prange(states.shape[0]):
        used = states[i].copy()
        rec(used, meta[i, 0], meta[i, 1], meta[i, 2], meta[i, 3], meta[i, 4], out[i])
    return out


def prefixes(n, depth):
    """Same branching rule as rec(), in Python, stopped after `depth` groups."""
    N = T * n
    jobs = []

    def go(used, left, lo, hi, k0, k2, placed):
        if left == 0 or placed == depth:
            jobs.append((used.copy(), (left, lo, hi, k0, k2)))
            return
        while used[lo]:
            lo += 1
        while used[hi]:
            hi -= 1
        a, b = n_start(used, lo, hi), n_end(used, lo, hi)
        if a == 0 or b == 0:
            return
        sign, base = (1, lo) if a <= b else (-1, hi)
        d = 1
        while (lo + 3 * d <= hi) if sign == 1 else (hi - 3 * d >= lo):
            cells = [base + sign * j * d for j in range(4)]
            if all(not used[c] for c in cells[1:]):
                for c in cells:
                    used[c] = 1
                r = d % 4
                go(used, left - 1, lo, hi, k0 + (r == 0), k2 + (r == 2), placed + 1)
                for c in cells:
                    used[c] = 0
            d += 1

    used = np.zeros(N + 2, np.uint8)
    used[0] = used[N + 1] = 1                     # sentinels so lo/hi scans stop
    go(used, n, 1, N, 0, 0, 0)
    return jobs


def count(n, depth, batch, ckpt=None):
    if n == 0:
        return np.ones((1, 1), np.int64), 0
    jobs = prefixes(n, depth)
    order = np.random.default_rng(1).permutation(len(jobs))
    hist = np.zeros((n + 1, n + 1), np.int64)
    done = set()
    if ckpt and os.path.exists(ckpt):
        for line in open(ckpt, encoding="utf-8"):
            rec_ = json.loads(line)
            if rec_["n"] == n and rec_["njobs"] == len(jobs):
                done.add(rec_["b"])
                hist += np.array(rec_["hist"], np.int64)
    nb = (len(jobs) + batch - 1) // batch
    t0 = time.time()
    for b in range(nb):
        if b in done:
            continue
        idx = order[b * batch:(b + 1) * batch]
        states = np.array([jobs[i][0] for i in idx])
        meta = np.array([jobs[i][1] for i in idx], np.int64)
        h = run_batch(states, meta, n).sum(axis=0)
        hist += h
        if ckpt:
            with open(ckpt, "a", encoding="utf-8") as f:
                f.write(json.dumps({"n": n, "njobs": len(jobs), "b": b, "hist": h.tolist()}) + "\n")
            print(f"  n={n} batch {b + 1}/{nb}  running total {int(hist.sum())}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return hist, len(jobs)


def show(n, hist):
    for k0 in range(n + 1):
        for k2 in range(n + 1 - k0):
            if hist[k0, k2]:
                print(f"    k0={k0:2d} k2={k2:2d} odd={n - k0 - k2:2d}  {int(hist[k0, k2])}")


if __name__ == "__main__":
    lo_n, hi_n = int(sys.argv[1]), int(sys.argv[2])
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    known = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 and sys.argv[4] else []
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
    for n in range(lo_n, hi_n + 1):
        t0 = time.time()
        ck = f"buckets_n{n}.jsonl" if n >= 18 else None
        hist, nj = count(n, depth, batch, ck)
        a = int(hist.sum())
        ok = "NEW" if n >= len(known) else ("ok" if a == known[n] else f"MISMATCH (known {known[n]})")
        print(f"t=4 n={n:2d}  a={a}  {ok}  jobs={nj}  {time.time() - t0:.1f}s", flush=True)
        show(n, hist)
