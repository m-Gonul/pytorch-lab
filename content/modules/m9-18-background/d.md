Stauffer–Grimson GMM (PyTorch, piksel başına `K` bileşen, gri seviye, `float64`).

- `class GMM(shape, K=3, alpha=0.01, var0=225.0, T=0.7, w0=0.05)` — durum: `w, mu, var` `(K, H, W)`; başlangıçta ağırlıklar 0 (ilk kare ilk bileşeni `w=1, mu=I_0` yapar).
- `apply(frame)` → ön plan maskesi `(H, W)` bool:
  1. Bileşenleri $w/\sigma$'ya göre azalan sırala (`argsort` + `gather`).
  2. Eşleşme: sıradaki ilk $|x-\mu|<2.5\sigma$ (ve $w>0$) bileşen.
  3. Arka plan bileşenleri: birikimli ağırlığı (sıralı) $T$'yi ilk geçene kadarki dahil; eşleşen bileşen arka plandaysa piksel arka plan.
  4. Güncelleme: $w\leftarrow(1-\alpha)w+\alpha M$; eşleşen için $\rho=\alpha$, $\mu\leftarrow\mu+\rho(x-\mu)$, $\sigma^2\leftarrow\sigma^2+\rho((x-\mu)^2-\sigma^2)$; eşleşme yoksa en son sıradaki bileşen ← $(w_0, x, \text{var0})$; ağırlıkları normalize et.

Kontrol, sentetik videoda F-ölçüsünü ViBe ve `cv2.createBackgroundSubtractorMOG2` ile karşılaştırır.
