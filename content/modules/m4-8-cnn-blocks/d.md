PyTorch.

- `DepthwiseSeparable(cin, cout, k=3)`, `SE(c, r=16)`, `Inception(cin, c1, c3r, c3, c5r, c5, pp)` (her dal Conv + ReLU; çıktılar kanal ekseninde birleşir).
- `ex5_models()` → `{"fc1", "cnn2", "fc3", "cnn4"}`: alıştırmanın dört modeli (28×28 girdi; cnn4: 7×7×64, BN, havuz, BN, 5×5×128, BN, 3×3×128, BN, havuz, BN, 3×3×256, BN, 3×3×256, BN, havuz, FC 128, 64, 10).
- `shifted_digits(n_rep, seed)` — sklearn rakamlarını 20×20'ye büyütüp 28×28 tuvalde rastgele konuma koy (Fashion-MNIST yerine çevrimdışı vekil; konum çeşitliliği CNN'in önselini sınar).
- `train_eval(model, X, y, Xt, yt, epochs=8)` → test doğruluğu (Adam 1e-3, batch 64).
