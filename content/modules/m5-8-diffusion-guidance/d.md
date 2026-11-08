PyTorch. Aynı 8 Gauss halkası, 4 sınıf; T = 200, $\beta$ 10⁻⁴'ten 0.1'e doğrusal.

- `EpsNet(C=4, h=128)` — girdi izdüşümü + sinüzoidal zaman embedding'i + sınıf embedding'i (C + 1 giriş: son indeks boş token), toplanıp SiLU'lu MLP'ye.
- `train(model, steps, p_uncond=0.15, seed=0)` — her batch'te etiketlerin `p_uncond`'u boş token'a çevrilir.
- `sample(model, n, y, s, seed)` — CFG'li deterministik DDIM (her adımda iki değerlendirme).
