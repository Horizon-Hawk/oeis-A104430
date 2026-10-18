"""Chunked, resumable GPU count. Usage: python gpu_run.py n [depth] [extra] [chunk] [budget] [known]"""
import sys
import time
from gpu_count import count_chunked

n = int(sys.argv[1])
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
extra = int(sys.argv[3]) if len(sys.argv) > 3 else 1
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 200000
budget = int(sys.argv[5]) if len(sys.argv) > 5 else 50000
known = [int(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else []
t0 = time.time()
a, nb = count_chunked(n, depth, extra, chunk, budget, f"gpu_n{n}.jsonl")
ok = "NEW" if n >= len(known) else ("ok" if a == known[n] else f"MISMATCH (known {known[n]})")
print(f"GPU n={n:2d}  a={a}  {ok}  base_jobs={nb}  {time.time() - t0:.1f}s", flush=True)
