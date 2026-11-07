Gauss karışım modeli (GMM) her noktanın gizli bir bileşenden $z\in\{1..K\}$ geldiğini varsayar:

$$
p(x)=\sum_{k=1}^K\pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k),\qquad
\ell(\theta)=\sum_i\log\sum_k\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k).
$$

Toplamın logaritması kapalı form MLE'yi engeller. **EM** iki kolay adımı tekrarlar:

- **E:** her nokta için bileşenlerin posterior olasılığı, [[responsibility]]: $r_{ik}=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}$.
- **M:** $r$'yi ağırlık sayarak her bileşen için ağırlıklı MLE:

$$
N_k=\sum_ir_{ik},\quad \pi_k=\frac{N_k}{n},\quad \mu_k=\frac{1}{N_k}\sum_ir_{ik}x_i,\quad \Sigma_k=\frac1{N_k}\sum_ir_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top .
$$

Sayısal olarak her şey log uzayında yapılır: $\log r_{ik}=\log\pi_k+\log\mathcal N_{ik}-\text{LSE}_j(\cdot)$.
