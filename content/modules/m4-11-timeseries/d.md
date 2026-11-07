PyTorch, ders alıştırması 6. `generate_time_series(n, steps, seed)` verilir (iki rastgele sinüs + gürültü; Géron). 5000 seri: 4000 eğitim, 1000 test; ilk 40 değer girdi.

- `ex6_models()` → `{1: doğrusal (40 → 1), 2: RNN(1), 3: RNN16 → RNN16 → RNN1, 4: RNN16 → RNN16 → yoğun 8 (son adım), 5: RNN16 → RNN16 → her adımda yoğun 8}`. Çıkışlar: modeller 1–3 `(N, 1)`, 4 `(N, 8)`, 5 `(N, 40, 8)`. RNN çıkışı son katmanın son adımı (model 2, 3).
- `seq_targets(S, n_steps=40, ahead=8)` — model 5 hedefi `(N, 40, 8)`: t adımında `S[t+1 … t+8]`.
- `train(m, X, Y, epochs=10, lr=1e-3)` — Adam, MSE, batch 32.
- `iterative_forecast(m, X, h, window=40)` — tek adımlık modeli kendi tahminiyle besle.
