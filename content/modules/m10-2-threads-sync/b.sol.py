from math import factorial


def _step(ins, regs, shared, locks, t):
    op = ins[0]
    if op == "load":
        regs = dict(regs)
        regs[ins[2]] = shared[ins[1]]
    elif op == "add":
        regs = dict(regs)
        regs[ins[1]] = regs.get(ins[1], 0) + ins[2]
    elif op == "store":
        shared = dict(shared)
        shared[ins[1]] = regs[ins[2]]
    elif op == "lock":
        if locks.get(ins[1]) is not None:
            return None
        locks = dict(locks)
        locks[ins[1]] = t
    elif op == "unlock":
        locks = dict(locks)
        locks[ins[1]] = None
    return regs, shared, locks


def explore(threads, init):
    n = len(threads)
    start = ((0,) * n, tuple(() for _ in range(n)), tuple(sorted(init.items())), ())
    seen, outcomes, deadlock = set(), set(), False
    stack = [start]
    while stack:
        st = stack.pop()
        if st in seen:
            continue
        seen.add(st)
        pcs, regs_t, sh_t, lk_t = st
        shared, locks = dict(sh_t), dict(lk_t)
        if all(pcs[i] == len(threads[i]) for i in range(n)):
            outcomes.add(sh_t)
            continue
        moved = False
        for i in range(n):
            if pcs[i] == len(threads[i]):
                continue
            r = _step(threads[i][pcs[i]], dict(regs_t[i]), shared, locks, i)
            if r is None:
                continue
            regs, sh2, lk2 = r
            moved = True
            pcs2 = pcs[:i] + (pcs[i] + 1,) + pcs[i + 1:]
            regs2 = regs_t[:i] + (tuple(sorted(regs.items())),) + regs_t[i + 1:]
            stack.append((pcs2, regs2, tuple(sorted(sh2.items())), tuple(sorted((k, v) for k, v in lk2.items() if v is not None))))
        if not moved:
            deadlock = True
    return outcomes, deadlock


def count_interleavings(lengths):
    out = factorial(sum(lengths))
    for L in lengths:
        out //= factorial(L)
    return out


inc = [("load", "x", "r"), ("add", "r", 1), ("store", "x", "r")]
print(explore([inc, inc], {"x": 0}))
