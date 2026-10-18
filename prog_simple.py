def a(n):
    N = 4 * n
    used = [False] * (N + 2)
    def go(left):
        if left == 0:
            return 1
        x = used.index(False, 1)   # smallest unused number starts a progression
        t, d = 0, 1
        while x + 3 * d <= N:
            if not (used[x + d] or used[x + 2 * d] or used[x + 3 * d]):
                for j in range(4): used[x + j * d] = True
                t += go(left - 1)
                for j in range(4): used[x + j * d] = False
            d += 1
        return t
    return go(n)
print([a(n) for n in range(12)])
