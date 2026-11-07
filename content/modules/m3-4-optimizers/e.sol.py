import math
import torch


@torch.no_grad()
def adamw_foreach_(params, grads, exp_avgs, exp_avg_sqs, step, lr, b1, b2, eps, wd):
    """Tüm listeleri yerinde güncelle. Parametreler üzerinde döngü yok."""
    torch._foreach_mul_(params, 1 - lr * wd)
    torch._foreach_mul_(exp_avgs, b1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - b1)
    torch._foreach_mul_(exp_avg_sqs, b2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - b2)
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, math.sqrt(bc2))
    torch._foreach_add_(denom, eps)
    torch._foreach_addcdiv_(params, exp_avgs, denom, value=-lr / bc1)
