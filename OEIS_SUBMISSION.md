# Draft edit for A104430 (paste into https://oeis.org/edit?seq=A104430)

Replace `[Name]` with your name exactly as registered on your OEIS account, and `https://github.com/Horizon-Hawk/oeis-A104430`
with the public repository link.

## DATA (append one term)
```
1,1,2,4,11,23,68,161,488,1249,3771,10388,35725,110449,387057,1411784,5938390,26054261,129231034,708657991,4216283527
```

## LINKS (add a line)
```
[Name], <a href="https://github.com/Horizon-Hawk/oeis-A104430">Programs and logs for a(20)</a> (CPU and GPU searches).
```

## PROG (add an entry)
```
(Python)
def a(n):
    N = 4*n
    used = [False]*(N+2)
    def go(left):
        if left == 0: return 1
        x = used.index(False, 1)  # smallest unused number starts a progression
        t, d = 0, 1
        while x + 3*d <= N:
            if not (used[x+d] or used[x+2*d] or used[x+3*d]):
                for j in range(4): used[x+j*d] = True
                t += go(left-1)
                for j in range(4): used[x+j*d] = False
            d += 1
        return t
    return go(n)
print([a(n) for n in range(12)])  # _[Name]_, Oct 18 2026
```

## EXTENSIONS (add a line)
```
a(20) from _[Name]_, Oct 18 2026
```

## Keywords
Leave `nonn,more` as is: more terms are still wanted.

## Notes for the "reason for edit" box
```
Added a(20) = 4216283527, computed by two independent implementations that agree exactly:
a multicore CPU backtracking search (4.5 h) and a separately written CUDA GPU search using
mirror symmetry (56 min). Both reproduce a(0)-a(19). Code and logs at the link.
```

---

## Second submission (optional, later): the all-odd-jump sequence
Proposed name: "Number of ways to split 1, 2, ..., 4n into n arithmetic progressions of 4 terms,
each with an odd common difference."
Data (n = 0..16): 1, 1, 1, 2, 4, 7, 12, 24, 44, 93, 186, 359, 821, 1815, 3656, 8452, 20394
Cross-ref: A104430. Not in the OEIS as of 2026-10-18. Before submitting, check a(0) = 1 by
convention (the empty split) and extend the data with `ap_buckets.py`.
