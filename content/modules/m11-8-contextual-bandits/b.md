Saf Python. **Kişiselleştirilmiş haber:** K = 5 makale, her birinin gizli $\theta_a\in[0,0.5]^5$; kullanıcı bağlamı $x$ 5 boyutlu, negatif olmayan ve toplamı 1 (ilgi dağılımı); tıklama olasılığı $\theta_a^\top x$.

- `class BayesLinReg(d, lam=1.0, sigma=1.0)` — `Sigma` (liste listesi) ve `b` ($\sigma^{-2}\sum xr$) tutar. `update(x, r)`: Sherman–Morrison ile $\Sigma\leftarrow\Sigma-\dfrac{(\Sigma x)(\Sigma x)^\top}{\sigma^2+x^\top\Sigma x}$. `mean(x)` = $x^\top\Sigma b$, `var(x)` = $x^\top\Sigma x$.
- `news(T, policy, seed=0, c=1.0)` → toplam tıklama. `policy`: `"linucb"` (makale başına ayrı model, $\sigma=0.5$), `"ucb1"` (bağlamsız), `"random"`.
