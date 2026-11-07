**Go:** iki oyuncu sırayla taş koyar; özgürlüğü kalmayan bağlı taşlar alınır; en çok alanı kontrol eden kazanır. Dallanma ~250, derinlik ~150: tam arama imkânsız. AlphaGo (DeepMind) Mart 2016'da Lee Sedol'u (9 dan), Mayıs 2017'de dünya şampiyonu Ke Jie'yi yendi.

**AlphaGo'nun dört aşaması:**

1. **Denetimli politika ağı** $p_\sigma(a|s)$: KGS'den 30 milyon konum; uzman hamlesi $a_i^*$ optimal sayılır, $\max\sum_i\log\pi_\sigma(a_i^*|s_i)$.
2. **RL politika ağı** $p_\rho$: eski sürümlerine karşı oynar; oyun sonu $G_n=\pm1$, γ = 1 ile REINFORCE: $\nabla=\sum_n\nabla\log\pi(a_n|s_n)G_n$.
3. **Değer ağı** $v_\theta(s)$: $p_\rho$'nun kendine karşı oyunlarından $(s,G)$ çiftlerine regresyon, $\tfrac12(G-v_\theta(s))^2$; korelasyonu kırmak için her oyundan bir konum.
4. **Arama:** her kenarda $Q(s,a)$, $P(s,a)$ (= $p_\sigma$), $N(s,a)$; seçim $a^*=\arg\max Q+u$, $u\propto P/(1+N)$; yaprakta $V=(1-\lambda)v_\theta(s_L)+\lambda z_L$ ($z_L$ hızlı rollout sonucu).

**AlphaGo Zero:** insan verisi yok, rollout yok, tek ağ $(p,v)=f_\theta(s)$. Kendine karşı oyunda her hamlede MCTS (800 benzetim); hamle $\pi(a)\propto N(s_0,a)^{1/\tau}$ ile seçilir (ilk 30 hamlede τ = 1, sonra → 0). Kökte $P=(1-\varepsilon)p+\varepsilon\,\mathrm{Dir}(\alpha)$. Oyun sonunda her konuma $z=\pm1$ (sıradaki oyuncunun bakışından). Kayıp:

$$\ell=(z-v)^2-\pi^\top\log p+c\|\theta\|^2.$$

Ağ π'yi taklit eder → daha iyi önsel → daha iyi arama: **arama politika iyileştirme operatörü**, $z$ politika değerlendirme operatörü. Simetriler (8) ile veri artırımı. Elo: $E_A=1/(1+10^{(R_B-R_A)/400})$.
