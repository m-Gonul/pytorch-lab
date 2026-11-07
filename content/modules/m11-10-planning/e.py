import math
import torch

LINES = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])


def winner(B):
    ...


def batched_playouts(board, player, n, g):
    ...


def mcts_move(board, player, n_sims, n_rollouts, g, c=1.4):
    ...


g = torch.Generator().manual_seed(0)
print(batched_playouts([0] * 9, 1, 10000, g).mean())
