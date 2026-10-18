# A104430: a(20) = 4216283527

[OEIS A104430](https://oeis.org/A104430): the number of ways to split 1, 2, ..., 4n into n
arithmetic progressions of 4 terms each. The OEIS entry lists a(0)..a(19). This repository computes
**a(20) = 4,216,283,527** and checks it two independent ways.

| n | a(n) | how |
|---|---|---|
| 0–19 | 1, 1, 2, 4, …, 708657991 | reproduced exactly by every program here |
| **20** | **4216283527** | CPU search (4.5 h) and GPU search (56 min), exact agreement |

## The search
The smallest unused number must be the first term of its progression, so the search places
progressions in order of their first terms, trying every common difference d that fits.

Dead-end cut (`ap_search.py`, `ap_par.py`, `gpu_count.py`): the largest unused number must be the
last term of some progression of unused numbers that starts at or above the smallest unused number.
If no such progression exists, the branch is abandoned.

Mirror symmetry (`ap_fast.py`, `gpu_count.py`): x → 4n+1−x maps solutions to solutions and swaps
d1 (the difference of the progression containing 1) with dN (the difference of the progression
containing 4n). Only solutions with d1 ≤ dN are counted: a = 2·#(d1 < dN) + #(d1 = dN).

## Files
| file | what it is |
|---|---|
| `prog_simple.py` | short plain-Python reference (the OEIS PROG entry), practical to about n = 12 |
| `ap_search.py` | single-threaded numba search with the dead-end cut |
| `ap_par.py` | multicore version: prefixes of the search are split into independent jobs (**run 1** for a(20)) |
| `ap_fast.py` | `ap_par.py` plus switchable mirror symmetry and an extra "lonely number" test (benchmarked: slower) |
| `gpu_count.py`, `gpu_run.py` | CUDA kernel via CuPy, 128-bit bitmask state, mirror symmetry, chunked and resumable (**run 2** for a(20)) |
| `ap_buckets.py` | a different algorithm (branches on whichever end is more constrained) that also tallies each solution by its jump classes mod 4 |
| `bucket_patterns.py` | checks relations among the mod-4 bucket counts |
| `results/` | logs of the runs |

## Reproduce
```
pip install -r requirements.txt
python ap_par.py 4 0 20 4 "$(cat known.txt)"        # CPU, prints ok / NEW for each n
python gpu_run.py 20                                 # GPU, NVIDIA card with CUDA 13 driver
```
Timings on an Intel Core Ultra 7 270K Plus (24 threads) and an NVIDIA RTX 5070:

| n | CPU `ap_par.py` | CPU `ap_fast.py` with symmetry | GPU `gpu_count.py` |
|---|---|---|---|
| 17 | 31.6 s | 19.9 s | 12.0 s |
| 18 | 227 s | 141 s | 43 s |
| 19 | 2126 s | — | — |
| 20 | 16310 s | — | 3369 s |

The GPU kernel runs each thread for a bounded number of steps, saves its stack and is relaunched,
so no single kernel trips the Windows GPU timeout (~2 s).

## Byproducts: counting by jump class mod 4
Classify each progression by its difference d: odd, ≡ 2 (mod 4) or ≡ 0 (mod 4). Let k0 and k2 be the
number of progressions with d ≡ 0 and d ≡ 2 (mod 4). For n = 1..16 (`results/buckets_n1-16.txt`):

* **k0 and k2 are always even.** Each residue class mod 4 contains exactly n numbers. A progression
  with odd d puts one term in each class, one with d ≡ 2 puts two terms in each of two classes of
  the same parity, and one with d ≡ 0 puts all four terms in a single class. Balancing the classes
  forces the d ≡ 2 progressions to pair up, so k2 is even, and then k0 is even as well.
* **No odd differences:** the count is a(n/2)² for even n and 0 for odd n. The even and odd numbers
  then split independently, each a scaled copy of 1..2n.
* **All differences ≡ 0 (mod 4):** the count is a(n/4)⁴ for n divisible by 4, and 0 otherwise.
* **All differences odd:** 1, 1, 2, 4, 7, 12, 24, 44, 93, 186, 359, 821, 1815, 3656, 8452, 20394
  (n = 1..16). This sequence is not in the OEIS as of 2026-10-18.

## License
MIT.
