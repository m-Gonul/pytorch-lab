def philosophers(n, ordered):
    progs = []
    for i in range(n):
        a, b = i, (i + 1) % n
        if ordered and b < a:
            a, b = b, a
        progs.append([("take", a), ("take", b), ("put", a), ("put", b)])
    start = ((0,) * n, (None,) * n)
    seen, stack, deadlock = set(), [start], False
    while stack:
        st = stack.pop()
        if st in seen:
            continue
        seen.add(st)
        pcs, forks = st
        if all(pc == 4 for pc in pcs):
            continue
        moved = False
        for i in range(n):
            if pcs[i] == 4:
                continue
            op, f = progs[i][pcs[i]]
            if op == "take":
                if forks[f] is not None:
                    continue
                nf = forks[:f] + (i,) + forks[f + 1:]
            else:
                nf = forks[:f] + (None,) + forks[f + 1:]
            moved = True
            stack.append((pcs[:i] + (pcs[i] + 1,) + pcs[i + 1:], nf))
        if not moved:
            deadlock = True
    return deadlock, len(seen)


print(philosophers(3, False), philosophers(3, True))
