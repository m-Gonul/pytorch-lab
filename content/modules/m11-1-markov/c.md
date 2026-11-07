NumPy, **döngü yok**.

- `n_step(T, p0, n)` — `np.linalg.matrix_power`.
- `stationary(T)` — `np.linalg.eig`, özdeğeri 1'e en yakın özvektör, gerçel kısım, toplam 1.
- `tv_curve(T, p0, n)` — $t=0..n$ için $\frac12\lVert\mathbf p_t-\boldsymbol\pi\rVert_1$ (toplam varyasyon uzaklığı); kuvvetleri tek seferde: `np.linalg.matrix_power` yerine özdeğer ayrışımı $T=V\Lambda V^{-1}$ ile $\mathbf p_t=V\Lambda^tV^{-1}\mathbf p_0$.
- `ar1_paths(phi, sigma, n, paths, seed=0)` — `(paths, n)`; `scipy.signal.lfilter([1], [1, -phi], w, axis=1)` ile döngüsüz.
- `ensemble_stats(X)` — her $t$ için topluluk ortalaması ve varyansı `(mean (n,), var (n,))`.
