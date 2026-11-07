**Üç aile:** Q-öğrenme (modelsiz, değer tabanlı; açık politika yok), politika gradyanı (modelsiz, politika tabanlı; açık değer yok), **aktör–eleştirmen** (ikisi birden).

**Rastgele politika** $\pi_\theta(a|s)=P(a|s;\theta)$: durumdan bir eylem değil, eylemler üzerinde dağılım verir. Ayrık: $\mathrm{softmax}(h_\theta(s,\cdot))$; sürekli: $N(a\,|\,\mu_\theta(s),\sigma_\theta(s)^2)$.

- **Taklit:** uzman çiftleri $(s_i,a_i^*)$ → $\max_\theta\sum_i\log\pi_\theta(a_i^*|s_i)$ (AlphaGo'nun ilk aşaması).
- **REINFORCE:** $\nabla_\theta\leftarrow\sum_n\gamma^nG_n\nabla\log\pi_\theta(a_n|s_n)$, $\theta\leftarrow\theta+\alpha\nabla_\theta$. Toplam indirimli getirisi büyük eylemlerin olasılığı artar.

**Politika gradyanı teoremi** (Sutton ve ark. 2000): $\nabla J(\theta)=\sum_s\mu_\theta(s)\sum_a\nabla\pi_\theta(a|s)\,Q_\theta(s,a)$; $\mu_\theta$ (indirimli) durum dağılımı. Durum dağılımının gradyanı gerekmez — kanıtın zor ve şaşırtıcı kısmı budur. Tek örnekli kestirim REINFORCE'u verir.

**Taban çizgisi:** $\sum_a\nabla\pi(a|s)\,(Q(s,a)-b(s))$ aynı gradyan, çünkü $\sum_a\nabla\pi=\nabla1=0$. $b(s)\approx V^\pi(s)$ → **avantaj** $A(s,a)=Q(s,a)-V(s)$: "bu durumda bu eylem ortalamadan ne kadar iyi". Politika ağı θ, değer ağı w.

- REINFORCE + taban çizgisi: $V_w$'yi MC ile öğren, $\delta=G_n-V_w(s_n)$.
- **Aktör–eleştirmen:** $V_w$'yi TD ile öğren, $\delta=r+\gamma V_w(s')-V_w(s)$ — düşük varyans, hızlı yakınsama (biraz yanlılık).
- **A2C:** n adımlı avantaj, paralel ortamlar, entropi bonusu.

**DPG** (Silver ve ark. 2014): sürekli eylem, deterministik $\mu_\theta(s)$: $\nabla J=\mathbb E_s[\nabla_\theta\mu_\theta(s)\nabla_aQ(s,a)|_{a=\mu_\theta(s)}]$. Keşif davranış gürültüsüyle (politika dışı), eleştirmen $Q_w$.
