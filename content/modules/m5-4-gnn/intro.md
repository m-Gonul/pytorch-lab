Bir grafın düğüm özellikleri $X\in\mathbb R^{N\times d}$, kenarları `edge_index` $\in\mathbb N^{2\times E}$ (satır 0 kaynak, satır 1 hedef; yönsüz graf için iki yön de listelenir).

**[[message passing]]:** her düğüm komşularından mesaj toplar ve kendini günceller:

$$
h_i'=\phi\Big(h_i,\ \bigoplus_{j\in\mathcal N(i)}\psi(h_i,h_j)\Big),\qquad \bigoplus\in\{\textstyle\sum,\text{ort},\max\}.
$$

**GCN** (Kipf & Welling) bunun doğrusal, normalize edilmiş hâli:

$$
H'=\sigma\big(\hat A\,H\,W\big),\qquad \hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2},\quad \tilde D_{ii}=1+\deg(i).
$$

Kenar başına bakınca: $\ h_i'=\sum_{j\in\mathcal N(i)\cup\{i\}}\frac{1}{\sqrt{\tilde d_i\tilde d_j}}\,(Wh_j)$. Bu toplam, kenar listesi üzerinde bir **scatter-add**'dir; yoğun $N\times N$ matris hiç kurulmaz.
