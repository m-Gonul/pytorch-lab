import torch


def random_sparse_mdp(S, A, k, seed=0):
    ...


def eval_direct(Pi_dense, r, gamma):
    ...


def eval_iterative(idx, prob, r, gamma, tol=1e-10):
    ...


def mpi_sparse(idx, prob, R, gamma, k, eps=1e-8):
    ...


idx, prob, R = random_sparse_mdp(100, 3, 5)
print(idx.shape, prob.shape, R.shape)
