"""Count splits of {1..t*n} into n t-term arithmetic progressions by direct search (numba-compiled).

Search: the lowest free number must START a progression; try every jump d that fits.
Dead-end cut: the HIGHEST free number must END some progression of free numbers that starts
at or above the current lowest free number -- if no jump works for it, the branch is hopeless.
Usage: python ap_search.py t n_from n_to [known,comma,list]
"""
import sys
import time
import numpy as np
from numba import njit


@njit(cache=True)
def top_ok(used, lo, N, t):
    z = N
    while z >= lo and used[z]:
        z -= 1
    if z < lo:
        return True                     # nothing left: fine
    d = 1
    while z - (t - 1) * d >= lo:
        ok = True
        for j in range(1, t):
            if used[z - j * d]:
                ok = False
                break
        if ok:
            return True
        d += 1
    return False


@njit(cache=True)
def search(t, n):
    N = t * n
    if n == 0:
        return 1
    used = np.zeros(N + 2, np.uint8)
    xs = np.zeros(n + 1, np.int64)      # start of the progression at each level
    ds = np.zeros(n + 1, np.int64)      # jump currently tried at each level
    count = 0
    level = 0
    xs[0] = 1
    ds[0] = 0
    while level >= 0:
        x = xs[level]
        d = ds[level]
        if d > 0:                       # undo the progression placed with this d
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


if __name__ == "__main__":
    t, lo, hi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    known = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else []
    for n in range(lo, hi + 1):
        t0 = time.time()
        a = search(t, n)
        ok = "NEW" if n >= len(known) else ("ok" if a == known[n] else f"MISMATCH (known {known[n]})")
        print(f"t={t} n={n:2d}  a={a}  {ok}  {time.time() - t0:.1f}s", flush=True)
