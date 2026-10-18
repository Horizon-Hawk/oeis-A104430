"""Parallel version of ap_search.py: enumerate the first few progressions as independent jobs,
then search each job's subtree on all cores (numba prange).
Usage: python ap_par.py t n_from n_to [prefix_depth] [known,comma,list]
"""
import sys
import time
import numpy as np
from numba import njit, prange
from ap_search import top_ok


@njit(cache=True)
def search_from(used, t, n, level0, x0):
    """Count completions from a partial state: `level0` progressions placed, lowest free number x0."""
    N = t * n
    if level0 == n:
        return 1
    xs = np.zeros(n + 1, np.int64)
    ds = np.zeros(n + 1, np.int64)
    count = 0
    level = level0
    xs[level] = x0
    ds[level] = 0
    while level >= level0:
        x = xs[level]
        d = ds[level]
        if d > 0:
            for j in range(t):
                used[x + j * d] = 0
        d += 1
        placed = False
        while x + (t - 1) * d <= N:
            ok = True
            for j in range(1, t):
                if used[x + j * d]:
                    ok = False
                    break
            if ok:
                for j in range(t):
                    used[x + j * d] = 1
                nx = x + 1
                while nx <= N and used[nx]:
                    nx += 1
                if level + 1 == n:
                    count += 1
                    for j in range(t):
                        used[x + j * d] = 0
                elif top_ok(used, nx, N, t):
                    ds[level] = d
                    level += 1
                    xs[level] = nx
                    ds[level] = 0
                    placed = True
                    break
                else:
                    for j in range(t):
                        used[x + j * d] = 0
            d += 1
        if not placed:
            ds[level] = 0
            level -= 1
    return count


@njit(parallel=True, cache=True)
def run_jobs(states, starts, levels, t, n):
    out = np.zeros(states.shape[0], np.int64)
    for i in prange(states.shape[0]):
        used = states[i].copy()
        out[i] = search_from(used, t, n, levels[i], starts[i])
    return out


def prefixes(t, n, depth):
    """All valid partial placements of the first `depth` progressions (with the dead-end cut)."""
    N = t * n
    jobs = []

    def go(used, level, x):
        if level == depth or level == n:
            jobs.append((used.copy(), x, level))
            return
        d = 1
        while x + (t - 1) * d <= N:
            if all(not used[x + j * d] for j in range(1, t)):
                for j in range(t):
                    used[x + j * d] = 1
                nx = x + 1
                while nx <= N and used[nx]:
                    nx += 1
                if level + 1 == n or top_ok(used, nx, N, t):
                    go(used, level + 1, nx)
                for j in range(t):
                    used[x + j * d] = 0
            d += 1

    go(np.zeros(N + 2, np.uint8), 0, 1)
    return jobs


def count(t, n, depth):
    if n == 0:
        return 1, 0
    jobs = prefixes(t, n, depth)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(jobs))                 # mix big and small subtrees
    states = np.array([jobs[i][0] for i in order])
    starts = np.array([jobs[i][1] for i in order], np.int64)
    levels = np.array([jobs[i][2] for i in order], np.int64)
    return int(run_jobs(states, starts, levels, t, n).sum()), len(jobs)


if __name__ == "__main__":
    t, lo, hi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    known = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else []
    for n in range(lo, hi + 1):
        t0 = time.time()
        a, nj = count(t, n, depth)
        ok = "NEW" if n >= len(known) else ("ok" if a == known[n] else f"MISMATCH (known {known[n]})")
        print(f"t={t} n={n:2d}  a={a}  {ok}  jobs={nj}  {time.time() - t0:.1f}s", flush=True)
