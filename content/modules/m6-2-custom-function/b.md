Saf Python. Hazır: `Ctx` (`save_for_backward`, `saved`) ve `Function.apply(*args) → (çıktı, ctx)`.

- `LogSumExp` — `forward(ctx, xs)` (liste → skaler, kararlı: max çıkar), `backward(ctx, g)` → liste.
- `RoundSTE` — `forward` eleman eleman `round`, `backward` gradyanı aynen geçirir.
- `gradcheck(fn_cls, xs, eps=1e-6, tol=1e-5)` → `(geçti_mi, en_büyük_hata)`. Skaler çıktılı fonksiyonlar için: `backward(ctx, 1.0)` ile analitik gradyan, merkezi farkla sayısal gradyan.
