Saf Python. `X` liste listesi $(N\times D)$, istatistikler sütun başına.

- `bn_forward(X, gamma, beta, eps=1e-5)` → `(out, cache)`
- `bn_backward(dout, cache)` → `(dX, dgamma, dbeta)`; intro'daki sadeleşmiş formül.
- `rms_norm(x, g, eps=1e-6)` — tek vektör.
