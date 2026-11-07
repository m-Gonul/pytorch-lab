import math
import torch


@torch.no_grad()
def adamw_foreach_(params, grads, exp_avgs, exp_avg_sqs, step, lr, b1, b2, eps, wd):
    """Tüm listeleri yerinde güncelle. Parametreler üzerinde döngü yok."""
    ...
