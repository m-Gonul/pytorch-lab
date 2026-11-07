import numpy as np


def warp_sectors(addr):
    ...


def warp_bank_degree(words):
    ...


def transpose_traffic(N, kind, T=32, pad=0):
    ...


print(transpose_traffic(256, "naive"), transpose_traffic(256, "tiled", pad=1))
