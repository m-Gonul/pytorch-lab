**Doğrusal olmayan ödül:** bağlam $x\sim U[-1,1]^2$; eylem 0 → 0.6, eylem 1 → $\mathbb 1[x_1x_2>0]$, eylem 2 → $\mathbb 1[x_1x_2<0]$ (XOR), gürültü N(0, 0.1²). Verildi: `xor_means`.

- `class BLR(K, d, lam, sigma)` — eylem başına Bayesçi doğrusal regresyon, toplu tensörler: kesinlik `P (K, d, d)`, `b (K, d)`; `update(a, z, r)`, `refit(Z, A, R)` (sıfırdan, tüm veriyle), `sample(z, g)` → `(K,)` örneklenmiş ödüller (Cholesky).
- `run_lints(T, seed=0)` — öznitelik $z=(x,1)$; kümülatif pişmanlık listesi.
- `run_neural_linear(T, seed=0, retrain=50, iters=300)` — gövde 2→50→50 (ReLU) + doğrusal başlık (K çıkış), Adam 1e-2; her `retrain` adımda gövde+başlığı gözlenen (x, a, r) üzerinde MSE ile eğit (mini-batch 64), sonra BLR'yi $z=(\text{gövde}(x),1)$ özniteliklerinden yeniden kur. Eylem: BLR'den Thompson. `lam=1, sigma=0.5`.
