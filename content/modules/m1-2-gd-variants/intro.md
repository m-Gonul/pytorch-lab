M1.1'deki gradient descent her adımda **tüm** veriyi kullanıyordu: $g=\nabla L(\theta)=\frac1n\sum_i\nabla\ell_i(\theta)$. $n$ milyonlarca olunca tek adım bile pahalı. Üç seçenek:

| yöntem | adımda kullanılan | adım maliyeti | gürültü |
|---|---|---|---|
| batch GD | $n$ örnek | $O(n)$ | yok |
| [[sgd|SGD]] | 1 örnek | $O(1)$ | yüksek |
| mini-batch | $B$ örnek (32–4096) | $O(B)$ | $\propto 1/B$ |

Bir [[epoch]], verinin tamamının bir kez görülmesi: $\lceil n/B\rceil$ adım. Pratikte "SGD" denince karıştırılmış mini-batch kastedilir.

[[momentum|Momentum]] gradyanların üstel hareketli ortalamasını tutar:

$$
v_t=\beta v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_t .
$$

Bu, PyTorch'un `SGD(momentum=β)` formülüdür (dampening = 0). Tutarlı yönlerde hız birikir, salınan yönler sönümlenir.
