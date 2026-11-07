Saf Python, liste listeleri:

- `softmax_row(xs)` — `-inf` girdileri destekle (max'ı çıkar; `math.exp(-inf) = 0`).
- `attention(Q, K, V, causal=False)` → `(T, d_v)` liste; $T\times d_k$ girdiler, ölçek $1/\sqrt{d_k}$.
