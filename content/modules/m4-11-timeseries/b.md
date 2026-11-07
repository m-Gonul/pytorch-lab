Saf Python.

- `difference(y, d=1)` — d tur fark.
- `undifference(y, dz, d)` — `y` gözlenmiş seri, `dz` d kez farklanmış serinin gelecek tahminleri → `y`'nin gelecek tahminleri (özyineleme: bir alt mertebeye in, sonra kümülatif topla).
- `arma_predict(y, alpha, theta)` — her t = 0..n için bir adımlık tahmin (n + 1 eleman, sonuncusu geleceğe); başlangıçtan önceki y ve ε sıfır.
- `ar_acf(phi, nlags)` — AR(p)'nin kuramsal otokorelasyonları $\rho_0..\rho_{nlags}$: önce $\rho_1..\rho_p$ için p × p Yule–Walker sistemini (Gauss eleme) çöz, sonra özyinele.
- `rnn_params(sizes, n_in=1, double_bias=False)` — yığılmış basit RNN katmanlarının parametre sayısı (Keras: tek bias; PyTorch: iki).
- `receptive_field(k, dilations)`.
