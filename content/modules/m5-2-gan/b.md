Saf Python, **Dirac GAN**. $f'(t)=\sigma(-t)$, dolayısıyla

$$
\partial_\theta V=\psi\,\sigma(-\psi\theta),\qquad \partial_\psi V=\theta\,\sigma(-\psi\theta).
$$

Üreteç $V$'yi küçültür, ayırıcı büyütür (eksi R1 cezası $\tfrac\gamma2\psi^2$: gerçek veride $\nabla_xD$ büyüklüğü $\psi$ ile orantılı).

- `grads(theta, psi)` → `(gθ, gψ)`.
- `simulate(theta, psi, h, steps, gamma=0.0, alternating=False)` → son `(theta, psi)`:
  - eşzamanlı: ikisi de **eski** değerlerle güncellenir: $\theta\leftarrow\theta-h\,g_\theta$, $\psi\leftarrow\psi+h(g_\psi-\gamma\psi)$.
  - `alternating=True`: önce $\theta$ güncellenir, $g_\psi$ **yeni** $\theta$ ile hesaplanır.

Kontrol üç davranışı sınar: eşzamanlı uzaklaşır, alternatif sınırlı kalır, R1 ile $(0,0)$'a yakınsar.
