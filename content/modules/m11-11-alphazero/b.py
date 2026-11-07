import math

LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]


def terminal(board):
    ...


def puct_search(board, player, net_fn, n_sims, c=1.5):
    ...


def visit_policy(N, tau):
    ...


def uniform_net(board, player):
    return [1 / 9] * 9, 0.0


print(puct_search([1, 1, 0, -1, -1, 0, 0, 0, 0], 1, uniform_net, 50))
