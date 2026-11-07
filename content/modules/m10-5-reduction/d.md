Triton (GPU yoksa yorumlayıcı).

- `block_sums(x, BLOCK=1024)` — her program bloğunu `tl.sum` ile indirger → kısmi toplamlar.
- `sum_two_pass(x)` — kısmi toplamlara yeniden uygula (tek blok kalana dek).
- `sum_atomic(x)` — her program kısmi toplamını `tl.atomic_add` ile tek hücreye ekler.
- `argmax(x)` — blok başına (max, indeks), sonra torch ile son indirgeme.
- `histogram(x, nbins)` — int32 girdiler, `tl.atomic_add(H + v, 1)`.
- `butterfly(x)` (PyTorch, Triton değil) — `x (n_warps, 32)`; `__shfl_xor_sync` taklidi: `for off in 16, 8, 4, 2, 1: x = x + x[:, lane ^ off]`; her şerit warp toplamını tutar.
