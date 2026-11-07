Üstel hareketli ortalama **doğrusal bir özyinelemedir**: $B_t=aB_{t-1}+bI_t$, $a=1-\alpha$, $b=\alpha$. Kapalı form: $B_t=a^t\big(B_0+\sum_{k=1}^{t}a^{-k}bI_k\big)$ → zaman ekseninde `cumsum`. Uzun videolarda $a^{-k}$ taşar; çözüm **parçalı tarama**: $L$'lik parçalarda yerel `cumsum`, parçalar arası tek bir taşıma (carry) — M10.6'daki paralel tarama fikrinin aynısı.

- `ema_scan(frames, alpha, L=64)` — `frames (T, H, W)` `float64` tensör → `(T, H, W)` $B_t$ ($B_0=I_0$); parça döngüsü (`T/L` tekrar) serbest, kare döngüsü yok.
- `ema_var_scan(frames, alpha, L=64)` — ayrıca $V_t=(1-\alpha)V_{t-1}+\alpha(I_t-B_{t-1})^2$, $V_0=\text{var0}=100$; `(B, V)`. (İpucu: önce B'yi tara, sonra $(I_t-B_{t-1})^2$ dizisini aynı tarayıcıyla.)
- `masks(frames, alpha, k=3.0)` — $|I_t-B_{t-1}|>k\sqrt{V_{t-1}}$, `t ≥ 1` (ilk kare 0).

Not: **seçici** güncelleme (yalnız arka plan piksellerini güncellemek) özyinelemeyi doğrusal olmayan yapar ve bu hileyi bozar.
