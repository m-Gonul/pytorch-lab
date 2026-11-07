LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]


def _term(b):
    for l in LINES:
        s = b[l[0]] + b[l[1]] + b[l[2]]
        if s == 3 or s == -3:
            return s // 3
    return 0 if all(b) else None


def minimax(board, player):
    count = [0]

    def rec(b, p):
        count[0] += 1
        t = _term(b)
        if t is not None:
            return t, None
        best = None
        for m in range(9):
            if b[m] == 0:
                b[m] = p
                v = rec(b, -p)[0]
                b[m] = 0
                if best is None or v * p > best[0] * p:
                    best = (v, m)
        return best

    v, m = rec(list(board), player)
    return v, m, count[0]


def alphabeta(board, player):
    count = [0]

    def rec(b, p, alpha, beta):
        count[0] += 1
        t = _term(b)
        if t is not None:
            return t, None
        best_m = None
        if p == 1:
            v = -2
            for m in range(9):
                if b[m] == 0:
                    b[m] = p
                    x = rec(b, -p, alpha, beta)[0]
                    b[m] = 0
                    if x > v:
                        v, best_m = x, m
                    alpha = max(alpha, v)
                    if alpha >= beta:
                        break
        else:
            v = 2
            for m in range(9):
                if b[m] == 0:
                    b[m] = p
                    x = rec(b, -p, alpha, beta)[0]
                    b[m] = 0
                    if x < v:
                        v, best_m = x, m
                    beta = min(beta, v)
                    if alpha >= beta:
                        break
        return v, best_m

    v, m = rec(list(board), player, -2, 2)
    return v, m, count[0]


print(minimax([0] * 9, 1)[::2], alphabeta([0] * 9, 1)[::2])
