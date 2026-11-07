Saf Python, `random` serbest.

- `dropout_forward(x, p, rng, train=True)` → `(y, mask)`; her eleman için **sırayla** bir `rng.random()` çağır, `mask_i = 1 if r >= p else 0`. `train=False` ise `(list(x), None)`.
- `dropout_backward(dy, mask, p)` → `dx`.
- `early_stopping(val_losses, patience, min_delta=0.0)` → `(best_epoch, stop_epoch)`; iyileşme: `loss < best - min_delta`. Hiç durmazsa `stop_epoch` son indeks.
