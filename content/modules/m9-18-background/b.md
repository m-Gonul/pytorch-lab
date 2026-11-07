Saf Python (liste listeleri; örnekler `samples[k][y][x]`).

- `vibe_classify(samples, frame, R=20, nmin=2)` — `|v − s| < R` olan örnek sayısı `≥ nmin` ise arka plan (0), değilse ön plan (1). Sayarken `nmin`'e ulaşınca dur (ViBe'nin hızlandırması; sonuç aynı).
- `running_average(frames, alpha)` — `B_0 = frames[0]`, `B_t = (1−α)B_{t−1} + αI_t`; tüm `B_t` listesi.
- `ema_masks(frames, alpha, T)` — `|I_t − B_{t−1}| > T` maskeleri (t ≥ 1).
