"""Look for relations among the mod-4 bucket counts (from buckets_small.txt)."""
import re
from collections import defaultdict

a = [int(v) for v in open("known.txt").read().strip().split(",")]
B = defaultdict(dict)                       # B[n][(k0, k2)] = count
n = None
for line in open("results/buckets_n1-16.txt", encoding="utf-8"):
    m = re.match(r"t=4 n=\s*(\d+)", line)
    if m:
        n = int(m.group(1))
        continue
    m = re.match(r"\s+k0=\s*(\d+) k2=\s*(\d+) odd=\s*(\d+)\s+(\d+)", line)
    if m:
        k0, k2, _, c = map(int, m.groups())
        B[n][(k0, k2)] = c

print("1) parity: any bucket with odd k0 or odd k2?",
      any(k0 % 2 or k2 % 2 for n in B for (k0, k2) in B[n]))

print("\n2) no odd jumps  vs  a(n/2)^2 ;  all jumps 0 mod 4  vs  a(n/4)^4")
for n in sorted(B):
    z = sum(c for (k0, k2), c in B[n].items() if k0 + k2 == n)
    f = a[n // 2] ** 2 if n % 2 == 0 else 0
    q = B[n].get((n, 0), 0)
    g = a[n // 4] ** 4 if n % 4 == 0 else 0
    print(f"   n={n:2d}  no-odd {z:>8} formula {f:>8} {'ok' if z == f else 'X'}   "
          f"all-0mod4 {q:>5} formula {g:>5} {'ok' if q == g else 'X'}")

print("\n3) share of solutions by number of odd jumps (n=16)")
tot = sum(B[16].values())
byodd = defaultdict(int)
for (k0, k2), c in B[16].items():
    byodd[16 - k0 - k2] += c
for o in sorted(byodd):
    print(f"   odd={o:2d}  {byodd[o]:>9}  {100 * byodd[o] / tot:5.1f}%")

print("\n4) all-odd-jump solutions c(n) and ratio")
c = [B[n].get((0, 0), 0) for n in sorted(B)]
print("   ", ",".join(map(str, c)))
for n in sorted(B):
    print(f"   n={n:2d}  all-odd {B[n].get((0, 0), 0):>8}  share {100 * B[n].get((0, 0), 0) / a[n]:5.1f}%")
