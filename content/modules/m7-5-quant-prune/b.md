Saf Python.

- `quantize_sym(xs, bits=8)` → `(q, s)`; tümü sıfırsa `s = 1`.
- `quantize_affine(xs, bits=8)` → `(q, s, z)`; $\min$'i $\le0$, $\max$'ı $\ge0$ olacak şekilde genişlet (0 tam temsil edilsin), sonra kırp.
- `dequantize(q, s, z=0)`.
- `magnitude_prune(ws, sparsity)` — $k=\lfloor\text{sparsity}\cdot n\rfloor$ en küçük $|w|$'yi sıfırla (eşitlikte küçük indeks önce budanır).
