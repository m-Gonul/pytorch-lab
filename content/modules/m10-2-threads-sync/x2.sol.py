def peterson_states(variant):
    def step(i, pc, flag, turn):
        j = 1 - i
        f = list(flag)
        if variant == "no_turn":
            if pc == 0:
                f[i] = 1
                return 1, tuple(f), turn
            if pc == 1:
                return (1 if f[j] else 2), flag, turn
            if pc == 2:
                return 3, flag, turn
            f[i] = 0
            return 4, tuple(f), turn
        first, second = (0, 1) if variant == "correct" else (1, 0)
        if pc in (0, 1):
            if (pc == 0 and first == 0) or (pc == 1 and second == 0):
                f[i] = 1
                return pc + 1, tuple(f), turn
            return pc + 1, flag, j
        if pc == 2:
            return (2 if (flag[j] and turn == j) else 3), flag, turn
        if pc == 3:
            return 4, flag, turn
        f[i] = 0
        return 5, tuple(f), turn

    cs = 2 if variant == "no_turn" else 3
    end = 4 if variant == "no_turn" else 5
    start = ((0, 0), (0, 0), 0)
    seen, stack = set(), [start]
    viol = deadlock = False
    while stack:
        st = stack.pop()
        if st in seen:
            continue
        seen.add(st)
        pcs, flag, turn = st
        if pcs[0] == cs and pcs[1] == cs:
            viol = True
        if pcs == (end, end):
            continue
        succ = []
        for i in (0, 1):
            if pcs[i] == end:
                continue
            npc, nf, nt = step(i, pcs[i], flag, turn)
            succ.append(((npc, pcs[1]) if i == 0 else (pcs[0], npc), nf, nt))
        if all(s == st for s in succ):
            deadlock = True
        stack.extend(succ)
    return viol, deadlock, len(seen)


print(peterson_states("correct"))
