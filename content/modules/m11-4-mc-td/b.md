Saf Python. **Rastgele yürüyüş:** durumlar 1..5 (A..E), 0 ve 6 terminal; C'den (3) başla, her adım ±1 eşit olasılıkla; 6'ya ulaşınca ödül 1, aksi halde 0; γ = 1. Gerçek değerler $i/6$.

- `walk_episode(rng)` — `random.Random` ile bir bölüm: `[(s, r, s2), …]` geçiş listesi.
- `mc(episodes, alpha, V0=0.5)` — her ziyaret, sabit α MC; her bölümden sonra RMS hatayı kaydet. `(V, rms_list)` — `V` 7 elemanlı (terminaller 0).
- `td0(episodes, alpha, V0=0.5)` — TD(0), her geçişte güncelle; aynı çıktı.
- `rms(V)` — 1..5 üzerinde gerçek değerlere RMS.
