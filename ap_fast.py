"""ap_par.py plus speedups, each switchable so they can be benchmarked one at a time.

  SYM    -- mirror symmetry. x -> N+1-x maps solutions to solutions and swaps
            d1 (jump of the group holding 1) with dN (jump of the group holding N).
            Count only d1 <= dN:  a = 2 * #(d1 < dN) + #(d1 == dN).
  LONELY -- every free number must still have a group of free numbers through it
            (in any of its 4 positions); otherwise the branch is dead.
Usage: python ap_fast.py n_from n_to sym lonely [depth] [known,comma,list]
"""
import sys
import time
import numpy as np
from numba import njit, prange

T = 4


@njit(cache=True)
def top_ok(used, lo, N, dmin_top):
    """Highest free number must END a group of free numbers starting >= lo.
    If that number is N itself, its jump must be >= dmin_top (symmetry)."""
    z = N
    while z >= lo and used[z]:
        z -= 1
    if z < lo:
        return True
    d = dmin_top if z == N else 1
    while z - 3 * d >= lo:
        if used[z - d] == 0 and used[z - 2 * d] == 0 and used[z - 3 * d] == 0:
            return True
        d += 1
    return False


@njit(cache=True)
def lonely_ok(used, lo, N):
    """Every free y in [lo, top] must lie on some free 4-term progression inside [lo, top]."""
    z = N
    while z >= lo and used[z]:
        z -= 1
    for y in range(lo + 1, z):                    # the two ends are checked elsewhere
        if used[y]:
            continue
        found = False
        for p in range(4):                        # y is member p of the group
            d = 1
            while y - p * d >= lo and y + (3 - p) * d <= z:
                s = y - p * d
                ok = True
                for j in range(4):
                    if j != p and used[s + j * d]:
                        ok = False
                        break
                if ok:
                    found = True
                    break
                d += 1
            if found:
                break
        if not found:
            return False
    return True


@njit(cache=True)
def search_from(used, n, level0, x0, d1, dN0, sym, lonely):
    """Returns (count with d1 < dN, count with d1 == dN); without sym everything is in the first slot."""
    N = T * n
    if level0 == n:                               # the prefix already finished it
        return (0, 1) if (sym and dN0 == d1) else (1, 0)
    xs = np.zeros(n + 1, np.int64)
    ds = np.zeros(n + 1, np.int64)
    lt = 0
    eq = 0
    level = level0
    xs[level] = x0
    ds[level] = 0
    dmin = d1 if sym else 1
    while level >= level0:
        x = xs[level]
        d = ds[level]
        if d > 0:
            for j in range(4):
                used[x + j * d] = 0
        d += 1
        placed = False
        while x + 3 * d <= N:
            if used[x + d] == 0 and used[x + 2 * d] == 0 and used[x + 3 * d] == 0:
                if x + 3 * d == N and d < dmin:   # group through N must have jump >= d1
                    d += 1
                    continue
                for j in range(4):
                    used[x + j * d] = 1
                nx = x + 1
                while nx <= N and used[nx]:
                    nx += 1
                if level + 1 == n:
                    if sym:
                        # find dN: the group that holds N. It ends at N with jump d at its level.
                        dN = dN0                  # set if the prefix already used N
                        for L in range(level0, level + 1):
                            dd = ds[L] if L != level else d
                            if xs[L] + 3 * dd == N:
                                dN = dd
                        if dN == d1:
                            eq += 1
                        else:
                            lt += 1
                    else:
                        lt += 1
                    for j in range(4):
                        used[x + j * d] = 0
                elif top_ok(used, nx, N, dmin) and (lonely == 0 or lonely_ok(used, nx, N)):
                    ds[level] = d
                    level += 1
                    xs[level] = nx
                    ds[level] = 0
                    placed = True
                    break
                else:
                    for j in range(4):
                        used[x + j * d] = 0
            d += 1
        if not placed:
            ds[level] = 0
            level -= 1
    return lt, eq


@njit(parallel=True, cache=True)
def run_jobs(states, meta, n, sym, lonely):
    out = np.zeros((states.shape[0], 2), np.int64)
    for i in prange(states.shape[0]):
        used = states[i].copy()
        lt, eq = search_from(used, n, meta[i, 0], meta[i, 1], meta[i, 2], meta[i, 3], sym, lonely)
        out[i, 0] = lt
        out[i, 1] = eq
    return out


def prefixes(n, depth, sym, lonely):
    N = T * n
    jobs = []

    def go(used, level, x, d1, dN):
        if level == depth or level == n:
            jobs.append((used.copy(), (level, x, d1, dN)))
            return
        dmin = d1 if (sym and level > 0) else 1
        d = 1
        while x + 3 * d <= N:
            if all(not used[x + j * d] for j in range(1, 4)):
                if level > 0 and x + 3 * d == N and d < dmin:
                    d += 1
                    continue
                for j in range(4):
                    used[x + j * d] = 1
                nx = x + 1
                while nx <= N and used[nx]:
                    nx += 1
                nd1 = d if level == 0 else d1
                ndN = d if x + 3 * d == N else dN
                if level + 1 == n or (top_ok(used, nx, N, nd1 if sym else 1)
                                      and (not lonely or lonely_ok(used, nx, N))):
                    go(used, level + 1, nx, nd1, ndN)
                for j in range(4):
                    used[x + j * d] = 0
            d += 1

    go(np.zeros(N + 2, np.uint8), 0, 1, 0, 0)
    return jobs


def count(n, depth, sym, lonely):
    if n == 0:
        return 1, 0
    jobs = prefixes(n, depth, sym, lonely)
    order = np.random.default_rng(0).permutation(len(jobs))
    states = np.array([jobs[i][0] for i in order])
    meta = np.array([jobs[i][1] for i in order], np.int64)
    out = run_jobs(states, meta, n, sym, lonely)
    lt, eq = int(out[:, 0].sum()), int(out[:, 1].sum())
    return (2 * lt + eq if sym else lt), len(jobs)


if __name__ == "__main__":
    lo, hi, sym, lonely = (int(v) for v in sys.argv[1:5])
    depth = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    known = [int(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else []
    for n in range(lo, hi + 1):
        t0 = time.time()
        a, nj = count(n, depth, sym, lonely)
        ok = "NEW" if n >= len(known) else ("ok" if a == known[n] else f"MISMATCH (known {known[n]})")
        print(f"sym={sym} lonely={lonely} n={n:2d}  a={a}  {ok}  jobs={nj}  {time.time() - t0:.1f}s",
              flush=True)
