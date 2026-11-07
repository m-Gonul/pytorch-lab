NumPy, **döngü yok**. `W: (out, in)`, `x: (B, in)`.

- `quant_per_tensor(W)` → `(q int8, s skaler)`; `quant_per_channel(W)` → `(q int8, s (out,))`.
- `int8_matmul(xq, sx, wq, sw)` → `(xq.astype(int32) @ wq.T.astype(int32)) * sx * sw` (`sw` skaler ya da `(out,)`).
- `rel_error(y, y_ref)` → $\lVert y-y_{ref}\rVert/\lVert y_{ref}\rVert$.
