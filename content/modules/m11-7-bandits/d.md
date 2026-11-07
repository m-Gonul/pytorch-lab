PyTorch, Bayesçi haydutlar.

- `normal_posterior(sum_r, n, mu0, var0, noise_var)` — Normal önsel, Normal olabilirlik (bilinen gürültü): $\text{kesinlik}=1/\sigma_0^2+n/\sigma^2$, ortalama $=(\mu_0/\sigma_0^2+\sum r/\sigma^2)/\text{kesinlik}$ → `(mean, var)` (tensörler, şekil korunur).
- `thompson(probs, T, n_runs, k=1, seed=0)` — Beta–Bernoulli, her kol için **k** sonsal örneğin ortalaması; deneyler paralel; ortalama kümülatif pişmanlık `(T,)`.
- `gaussian_thompson(means, T, n_runs, noise_sd=1.0, seed=0)` — Normal–Normal (önsel N(0, 1)); pişmanlık `(T,)`.
