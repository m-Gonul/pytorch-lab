Bir karar ağacı, girdi uzayını eksenlere paralel dikdörtgenlere böler ve her bölgede sabit bir tahmin verir. Her iç düğüm bir soru sorar: $x_j\le t$ mi?

Bir bölünmenin iyiliği, çocuklarının **safsızlığıyla** ölçülür. Sınıf oranları $p_c$ için:

$$
\text{Gini: }G=1-\sum_c p_c^2,\qquad \text{Entropi: }H=-\sum_c p_c\log_2p_c,
\qquad
\text{skor}(j,t)=\frac{n_L}{n}G_L+\frac{n_R}{n}G_R .
$$

**CART** algoritması açgözlüdür: her düğümde tüm $(j,t)$ çiftleri arasından skoru en küçük olanı seçer, sonra iki çocukta aynı şeyi özyinelemeli yapar. Aday eşikler, bir özelliğin sıralı farklı değerlerinin orta noktalarıdır.

Ağaçta gradyan yok. Ama sert soru $[x_j\le t]$'yi yumuşak bir kapıyla, $\sigma\big((w^\top x+b)/\tau\big)$, değiştirince model türevlenebilir hâle gelir ve autograd ile eğitilebilir ("soft decision tree"). d aşaması bu köprüyü kuruyor.
