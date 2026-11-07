def checkpoint_cost(sizes, k):
    ...


def best_k(sizes):
    ...


print(checkpoint_cost([1] * 16, 4))   # (7, 12)
print(checkpoint_cost([1] * 16, 1))   # (16, 0)
print(best_k([1] * 16))               # 3 (k = 3 ve 4 tepe 7 ile berabere; 3 daha az yeniden hesap)
