"""GPU (CUDA via CuPy) count of splits of {1..4n} into 4-term progressions.

Same rules as ap_fast.py with mirror symmetry (lowest free number starts a group; the highest
free number must end one; count d1 <= dN, a = 2*#(d1<dN) + #(d1==dN)), re-implemented with
128-bit bitmasks, one search job per GPU thread.
Windows resets the GPU if one kernel runs more than ~2 s, so every thread runs at most `budget`
steps, saves its stack to GPU memory and the kernel is relaunched until all jobs are finished.
Usage: python gpu_count.py n_from n_to [depth] [budget] [known,comma,list]
"""
import sys
import time
import numpy as np
import cupy as cp
from ap_fast import prefixes

MAXL = 32
SRC = r"""
typedef unsigned long long u64;
#define MAXL 32
__device__ __forceinline__ int isfree(const u64* w, int b) { return !((w[b >> 6] >> (b & 63)) & 1ull); }
__device__ __forceinline__ void flip(u64* w, int b) { w[b >> 6] ^= 1ull << (b & 63); }

__device__ int nextfree(const u64* w, int x) {           // sentinels guarantee termination
    for (int k = x >> 6; k < 2; k++) {
        u64 f = ~w[k];
        if (k == (x >> 6)) f &= (~0ull) << (x & 63);
        if (f) return (k << 6) + __ffsll((long long)f) - 1;
    }
    return 128;
}
__device__ int highfree(const u64* w) {
    for (int k = 1; k >= 0; k--) { u64 f = ~w[k]; if (f) return (k << 6) + 63 - __clzll((long long)f); }
    return -1;
}
__device__ int top_ok(const u64* w, int lo, int N, int dmin) {
    int z = highfree(w);
    if (z < lo) return 1;
    int d = (z == N) ? dmin : 1;
    while (z - 3 * d >= lo) {
        if (isfree(w, z - d) && isfree(w, z - 2 * d) && isfree(w, z - 3 * d)) return 1;
        d++;
    }
    return 0;
}

extern "C" __global__ void step(const int* act, int nact, int n, long long budget,
        u64* W, int* XS, int* DS, int* DNS, int* LEV, const int* LEV0, const int* D1,
        u64* LT, u64* EQ, int* DONE) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nact) return;
    int i = act[t];
    int N = 4 * n;
    u64 w[2] = {W[2 * i], W[2 * i + 1]};
    int xs[MAXL], ds[MAXL], dns[MAXL];
    int level = LEV[i], level0 = LEV0[i], d1 = D1[i];
    for (int L = level0; L <= level; L++) { xs[L] = XS[i * MAXL + L]; ds[L] = DS[i * MAXL + L]; dns[L] = DNS[i * MAXL + L]; }
    u64 lt = LT[i], eq = EQ[i];
    long long steps = 0;
    while (level >= level0 && steps < budget) {
        steps++;
        int x = xs[level], d = ds[level];
        if (d > 0) for (int j = 0; j < 4; j++) flip(w, x + j * d);
        d++;
        int placed = 0;
        while (x + 3 * d <= N) {
            if (isfree(w, x + d) && isfree(w, x + 2 * d) && isfree(w, x + 3 * d)) {
                int top = x + 3 * d;
                if (top == N && d < d1) { d++; continue; }
                for (int j = 0; j < 4; j++) flip(w, x + j * d);
                int dN = (top == N) ? d : dns[level];
                int nx = nextfree(w, x + 1);
                if (level + 1 == n) {
                    if (dN == d1) eq++; else lt++;
                    for (int j = 0; j < 4; j++) flip(w, x + j * d);
                } else if (top_ok(w, nx, N, d1)) {
                    ds[level] = d; level++; xs[level] = nx; ds[level] = 0; dns[level] = dN;
                    placed = 1;
                    break;
                } else {
                    for (int j = 0; j < 4; j++) flip(w, x + j * d);
                }
            }
            d++;
        }
        if (!placed) { ds[level] = 0; level--; }
    }
    W[2 * i] = w[0]; W[2 * i + 1] = w[1];
    for (int L = level0; L <= level; L++) { XS[i * MAXL + L] = xs[L]; DS[i * MAXL + L] = ds[L]; DNS[i * MAXL + L] = dns[L]; }
    LEV[i] = level; LT[i] = lt; EQ[i] = eq;
    if (level < level0) DONE[i] = 1;
}
"""
KERNEL = cp.RawKernel(SRC, "step")


def to_words(used, N):
    w = [0, 0]
    for b in range(128):
        if b == 0 or b > N or (b <= N and used[b]):
            w[b >> 6] |= 1 << (b & 63)
    return w


def expand(n, job, extra):
    """Split one prefix job `extra` more levels, same rules as ap_fast.prefixes."""
    from ap_fast import top_ok
    N = 4 * n
    out = []
    used0, (level0, x0, d10, dN0) = job

    def go(used, level, x, d1, dN, left):
        if left == 0 or level == n:
            out.append((used.copy(), (level, x, d1, dN)))
            return
        d = 1
        while x + 3 * d <= N:
            if not used[x + d] and not used[x + 2 * d] and not used[x + 3 * d]:
                if x + 3 * d == N and d < d1:
                    d += 1
                    continue
                for j in range(4):
                    used[x + j * d] = 1
                nx = x + 1
                while nx <= N and used[nx]:
                    nx += 1
                ndN = d if x + 3 * d == N else dN
                if level + 1 == n or top_ok(used, nx, N, d1):
                    go(used, level + 1, nx, d1, ndN, left - 1)
                for j in range(4):
                    used[x + j * d] = 0
            d += 1

    go(used0.copy(), level0, x0, d10, dN0, extra)
    return out


def count_chunked(n, depth, extra, chunk, budget, ckpt):
    """Depth-`depth` jobs, cut into chunks; each chunk is split `extra` more levels and run on
    the GPU. Each finished chunk is appended to `ckpt` (resume-safe)."""
    import json
    import os
    base = prefixes(n, depth, 1, 0)
    order = np.random.default_rng(2).permutation(len(base))
    nch = (len(base) + chunk - 1) // chunk
    done, total = {}, 0
    if os.path.exists(ckpt):
        for line in open(ckpt, encoding="utf-8"):
            r = json.loads(line)
            if r["n"] == n and r["nbase"] == len(base):
                done[r["c"]] = r["a"]
    t0 = time.time()
    for c in range(nch):
        if c in done:
            total += done[c]
            continue
        jobs = [j for i in order[c * chunk:(c + 1) * chunk] for j in expand(n, base[i], extra)]
        a, _, nl = count(n, depth, budget, jobs=jobs)
        total += a
        with open(ckpt, "a", encoding="utf-8") as f:
            f.write(json.dumps({"n": n, "nbase": len(base), "c": c, "a": a, "jobs": len(jobs)}) + "\n")
        print(f"  n={n} chunk {c + 1}/{nch}  jobs={len(jobs)}  launches={nl}  running total {total}  "
              f"{time.time() - t0:.0f}s", flush=True)
    return total, len(base)


def count(n, depth, budget, verbose=False, jobs=None):
    if n == 0:
        return 1, 0, 0
    N = 4 * n
    if jobs is None:
        jobs = prefixes(n, depth, 1, 0)
    lt0 = eq0 = 0
    rows = []
    for used, (level, x, d1, dN) in jobs:
        if level == n:                                 # finished inside the prefix
            if dN == d1:
                eq0 += 1
            else:
                lt0 += 1
        else:
            rows.append((used, level, x, d1, dN))
    m = len(rows)
    if m == 0:
        return 2 * lt0 + eq0, len(jobs), 0
    bits = np.ones((m, 128), np.uint8)                # bit 0 and bits > N stay set (sentinels)
    bits[:, 1:N + 1] = np.array([r[0] for r in rows], np.uint8)[:, 1:N + 1]
    W = cp.asarray(np.packbits(bits, axis=1, bitorder="little").view(np.uint64).reshape(-1))
    XS = np.zeros((m, MAXL), np.int32)
    DNS = np.zeros((m, MAXL), np.int32)
    lev = np.array([r[1] for r in rows], np.int32)
    XS[np.arange(m), lev] = [r[2] for r in rows]
    DNS[np.arange(m), lev] = [r[4] for r in rows]
    XS, DNS = cp.asarray(XS), cp.asarray(DNS)
    DS = cp.zeros((m, MAXL), cp.int32)
    LEV = cp.asarray(lev)
    LEV0 = cp.asarray(lev.copy())
    D1 = cp.asarray(np.array([r[3] for r in rows], np.int32))
    LT = cp.zeros(m, cp.uint64)
    EQ = cp.zeros(m, cp.uint64)
    DONE = cp.zeros(m, cp.int32)
    act = cp.arange(m, dtype=cp.int32)
    launches = 0
    while act.size:
        nb = (act.size + 127) // 128
        KERNEL((nb,), (128,), (act, np.int32(act.size), np.int32(n), np.int64(budget),
                               W, XS, DS, DNS, LEV, LEV0, D1, LT, EQ, DONE))
        cp.cuda.Device().synchronize()
        launches += 1
        act = cp.nonzero(DONE == 0)[0].astype(cp.int32)
        if verbose and launches % 50 == 0:
            print(f"    launch {launches}: {act.size} jobs still running", flush=True)
    lt = int(LT.sum().get()) + lt0
    eq = int(EQ.sum().get()) + eq0
    return 2 * lt + eq, len(jobs), launches


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    known = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else []
    for n in range(lo, hi + 1):
        t0 = time.time()
        a, nj, nl = count(n, depth, budget, verbose=n >= 18)
        ok = "NEW" if n >= len(known) else ("ok" if a == known[n] else f"MISMATCH (known {known[n]})")
        print(f"GPU n={n:2d}  a={a}  {ok}  jobs={nj}  launches={nl}  {time.time() - t0:.1f}s", flush=True)
