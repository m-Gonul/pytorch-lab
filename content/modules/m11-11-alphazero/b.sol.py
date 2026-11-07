import math

LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]


def terminal(board):
    for l in LINES:
        s = board[l[0]] + board[l[1]] + board[l[2]]
        if s == 3 or s == -3:
            return s // 3
    return 0 if all(board) else None


class _Node:
    def __init__(self, board, player, net_fn):
        self.board, self.player = board, player
        prior, self.value = net_fn(board, player)
        legal = [i for i in range(9) if board[i] == 0]
        tot = sum(prior[i] for i in legal) or 1.0
        self.P = [prior[i] / tot if board[i] == 0 else 0.0 for i in range(9)]
        self.N, self.W = [0] * 9, [0.0] * 9
        self.children, self.legal = {}, legal


def puct_search(board, player, net_fn, n_sims, c=1.5):
    root = _Node(list(board), player, net_fn)
    for _ in range(n_sims):
        node, path = root, []
        while True:
            sq = math.sqrt(sum(node.N) + 1)
            a = max(node.legal, key=lambda i: (node.W[i] / node.N[i] if node.N[i] else 0.0) + c * node.P[i] * sq / (1 + node.N[i]))
            path.append((node, a))
            if a in node.children:
                node = node.children[a]
                continue
            b2 = node.board[:]
            b2[a] = node.player
            t = terminal(b2)
            if t is not None:
                v = -abs(t)
            else:
                child = _Node(b2, -node.player, net_fn)
                node.children[a] = child
                v = child.value
            break
        for nd, a in reversed(path):
            v = -v
            nd.N[a] += 1
            nd.W[a] += v
    return root.N


def visit_policy(N, tau):
    if tau == 0:
        k = max(range(len(N)), key=lambda i: N[i])
        return [1.0 if i == k else 0.0 for i in range(len(N))]
    w = [n ** (1 / tau) for n in N]
    s = sum(w)
    return [x / s for x in w]


def uniform_net(board, player):
    return [1 / 9] * 9, 0.0


print(puct_search([1, 1, 0, -1, -1, 0, 0, 0, 0], 1, uniform_net, 50))
