import math
import torch

LINES = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])


def winner(B):
    s = B[:, LINES].sum(-1)
    return torch.where((s == 3).any(1), 1, torch.where((s == -3).any(1), -1, 0))


def batched_playouts(board, player, n, g):
    B = torch.tensor(board).repeat(n, 1)
    p = torch.full((n,), player)
    res = winner(B)
    done = (res != 0) | (B != 0).all(1)
    for _ in range(9):
        if bool(done.all()):
            break
        w = torch.where(done[:, None], torch.ones(n, 9), (B == 0).float())
        mv = torch.multinomial(w, 1, generator=g).squeeze(1)
        act = ~done
        B[act, mv[act]] = p[act]
        nw = winner(B)
        res = torch.where(act, nw, res)
        done = done | (nw != 0) | (B != 0).all(1)
        p = -p
    return res.double()


def _terminal(board):
    w = int(winner(torch.tensor([board]))[0])
    if w != 0:
        return w
    if all(v != 0 for v in board):
        return 0
    return None


class _Node:
    def __init__(self, board, player):
        self.board, self.player = board, player
        self.children, self.N, self.W = {}, 0, 0.0
        self.moves = [i for i in range(9) if board[i] == 0]


def mcts_move(board, player, n_sims, n_rollouts, g, c=1.4):
    root = _Node(list(board), player)
    for _ in range(n_sims):
        node, path = root, [root]
        while True:
            tv = _terminal(node.board)
            if tv is not None:
                value = float(tv)
                break
            if len(node.children) < len(node.moves):
                m = next(m for m in node.moves if m not in node.children)
                b = node.board[:]
                b[m] = node.player
                child = _Node(b, -node.player)
                node.children[m] = child
                path.append(child)
                tv = _terminal(b)
                value = float(tv) if tv is not None else float(batched_playouts(b, -node.player, n_rollouts, g).mean())
                break
            parent = node
            node = max(parent.children.values(), key=lambda ch: ch.W / ch.N * parent.player + c * math.sqrt(math.log(parent.N) / ch.N))
            path.append(node)
        for nd in path:
            nd.N += 1
            nd.W += value
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]


g = torch.Generator().manual_seed(0)
print(batched_playouts([0] * 9, 1, 10000, g).mean())
