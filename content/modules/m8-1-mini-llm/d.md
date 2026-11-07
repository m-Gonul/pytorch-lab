`GPT` hazır (M4.7'deki yapı, küçük başlatmalı). Sen yaz:

- `lr_at(step, warmup, total, max_lr, min_lr)` — b'deki program.
- `train(model, D, steps=1500, batch=128, max_lr=3e-3, seed=0)` — `torch.Generator(seed)` ile rastgele satırlar; `x = D[ix, :-1]`, `y = D[ix, 1:]`; kayıp yalnız `[:, 5:8]` konumlarında; AdamW (`weight_decay=0.01`), warmup 100 adım, `min_lr = max_lr / 10`, `clip_grad_norm_(…, 1.0)`. Son kaybı döndür.
- `solve(model, prompts)` — `prompts: (B, 6)` (`ab+cd=`), 3 adım greedy, `(B, 3)` cevap token'ları. Batch üzerinde vektörel.
- `accuracy(model, D)` — 3 basamağın tamamı doğru olan satırların oranı.
