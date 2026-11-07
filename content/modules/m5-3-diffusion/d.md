PyTorch, 8 Gauss'lu halka üzerinde DDPM.

- `make_schedule(T=100, beta1=1e-4, betaT=0.05)` → `(betas, abar)` (doğrusal).
- `TimeEmbedding(dim)` — sinüzoidal: frekanslar $f_k=\exp(-\log(1000)\,k/(\text{dim}/2))$, çıktı $[\sin(tf),\cos(tf)]$.
- `EpsNet(hidden=128, t_dim=32)` — `forward(x, t)`; `t` uzun tamsayı tensörü `(B,)`. Girdi `cat([x, emb(t)])`, üç gizli katman `SiLU`.
- `ddpm_loss(model, x0, abar, t=None, eps=None)` — verilmezse `t ~ U{0..T-1}`, `eps ~ N(0, I)`; ortalama kare hata.
- `ddpm_sample(model, betas, abar, n)` → `(n, 2)`; `@torch.no_grad()`.

Kontrol önce örnekleyiciyi kâhinle (eğitimsiz) sınar, sonra 3000 adım eğitip halkanın 8 modunun üretildiğini ölçer.
