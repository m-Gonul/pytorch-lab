Rakamlar (64 piksel). Verilenler: küçük bir otomatik kodlayıcı (8 boyutlu gizli kod) ve rakam sınıflandırıcısı.

- `CondDenoiser(D, C=10, h=256)` — `EpsNet`'in D boyutlu, 10 sınıflı sürümü.
- `train_diffusion(Z, y, steps, seed)` ve `sample(model, labels, D, s, seed)`.
- Karşılaştırma: piksel uzayı (D = 64, [−1, 1]) vs gizli uzay (D = 8, standartlaştırılmış kodlar, sonra kod çözücü) — aynı adım bütçesi, sınıf doğruluğu (sınıflandırıcıyla), güven, süre.
