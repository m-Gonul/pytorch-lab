NumPy.

- `intensity(flops, bytes_)` — eleman eleman (dizi destekli).
- `attainable(peak, bw, I)` → $\min(\text{peak}, \text{bw}\cdot I)$; döngüsüz.
- `op_costs(n)` → `{ad: (flop, bayt)}`, fp32, her tensör bir kez okunup/yazılır:
  - `"add"`: $n$ elemanlı iki vektör toplanır.
  - `"softmax"`: $n\times n$ satır softmax'ı, FLOP $\approx5n^2$, okuma + yazma.
  - `"matmul"`: $n\times n$ iki matris, FLOP $2n^3$, üç matris.
- `matmul_blocked(A, B, T)` — üç seviye tile döngüsü (i, j, k blokları), içte NumPy alt-matris çarpımı.
