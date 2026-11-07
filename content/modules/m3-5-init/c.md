- `xavier_std(fan_in, fan_out)`, `kaiming_std(fan_in)` — standart sapma (varyansın kökü).
- `activation_stds(X, depth, width, std, act, rng)` → uzunluğu `depth` olan liste: her katmandan sonra aktivasyonların std'si. Katman $l$: `W = rng.standard_normal((width, width)) * std`, `H = act(H @ W)`; `act` bir fonksiyon (`np.tanh`, ReLU…). Yalnız katman döngüsü.

Kontrol 30 katmanda üç senaryoyu dener ve tablo üretir: std = 1 (patlama), Xavier + ReLU (sönme), Kaiming + ReLU (kararlı).
