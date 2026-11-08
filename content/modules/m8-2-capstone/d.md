PyTorch. Veri: 16 × 16'ya büyütülmüş rakamlar, [−1, 1]; T = 200, β 10⁻⁴'ten 0.05'e doğrusal. Verilenler: `digits16`, `temb`, `AB`.

- `ResBlock(ci, co, e)`, `Attn(c)`, `UNet(c=16, C=10)` — intro'daki mimari (GroupNorm(4, ·)); `forward(x, t, y)`; boş token indeksi C.
- `train(model, steps, seed, ema_decay=0.99, p_uncond=0.1)` → `(model, ema)` — AdamW 2e-3, batch 64; her adımdan sonra EMA güncellemesi.
- `sample(net, labels, s, steps, seed)` — CFG'li deterministik DDIM; her adımda $\hat x_0$ [−1, 1]'e kırpılır (dinamik olmayan eşikleme).
