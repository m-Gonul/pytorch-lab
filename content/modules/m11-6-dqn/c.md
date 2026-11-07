**Baird karşı örneği** (Sutton Örnek 11.1), NumPy, döngü yalnız yineleme. 7 durum, 8 ağırlık: $\hat v(s_i)=2w_i+w_8$ (i = 1..6), $\hat v(s_7)=w_7+2w_8$. Tüm ödüller 0 (gerçek değerler 0, $w=0$ ile temsil edilebilir). Hedef politika hep "düz" eylemi seçer → $s_7$. Beklenen güncellemeler (dinamik programlama gibi, örnekleme yok), durum ağırlıkları $\mu$:

- `baird_features()` → `(7, 8)`.
- `expected_td(w0, alpha, gamma, n_iter, mu)` → `(w, normlar)`: $w\leftarrow w+\alpha\sum_s\mu_s\,\delta_s\,x(s)$, $\delta_s=\gamma\hat v(s_7)-\hat v(s)$.
- `expected_tdc(w0, alpha, beta, gamma, n_iter, mu)` — gradyan-TD (TDC): ikinci ağırlık $u$ ($u_0=0$):
  $w\leftarrow w+\alpha\sum_s\mu_s[\delta_sx_s-\gamma x_7(x_s^\top u)]$, $u\leftarrow u+\beta\sum_s\mu_s(\delta_s-x_s^\top u)x_s$.

Davranış dağılımı düzgün ($\mu=1/7$) → TD ıraksar; politika üstü ($\mu=e_7$) → kararlı; TDC düzgün dağılımda bile kararlı.
