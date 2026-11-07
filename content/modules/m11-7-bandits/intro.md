**Haydut:** tek durum $S=\{s\}$, eylemler (kollar) $A$, ödüller (genelde [0,1]'e ölçekli). Geçiş yok; yalnız rastgele ödül fonksiyonu öğrenilir. Ad: kumar makinesi ("tek kollu haydut"). Uygulamalar: klinik deney tasarımı, çevrimiçi reklam (tıklama oranı), web kişiselleştirme, oyunlar, ağ yönlendirme. Basit tanımlı ama genel durumda bilinen izlenebilir optimal çözüm yok.

**Pişmanlık:** $r^*=\max_aR(a)$; $\mathrm{loss}(a)=r^*-R(a)$; $\mathrm{Loss}_n=\sum_{t=1}^n\mathbb E[r^*-R(a_t)]$.

- Açgözlü: takılabilir. ε-açgözlü sabit ε: $P(a_t\ne a^*)\approx\varepsilon$ → $\mathrm{Loss}_n=O(n)$ (doğrusal). $\varepsilon_t\propto1/t$: $\sum1/t\approx\ln n+0.5772$ → $O(\log n)$.
- **İyimserlik:** her kol için gerçeği yüksek olasılıkla aşan bir üst sınır $\mathrm{UB}_n(a)$ al ve en büyüğünü seç. Sınır gerçeğe yakınsıyorsa, alt-optimal bir kola yakınsamak çelişki doğurur. Hoeffding: $P(R(a)\le\tilde R(a)+c)\ge1-e^{-2n_ac^2}$; $\delta_n=1/n^4$ ile $c=\sqrt{2\ln n/n_a}$ → **UCB1**, $O(\log n)$ pişmanlık.
- **Bayesçi:** $\theta$ üzerinde önsel, gözlemlerle sonsal $P(\theta|r_{1:n})\propto P(\theta)\prod P(r_i|\theta)$. Eşlenik çiftlerde (Beta–Bernoulli, Normal–Normal) sonsal kapalı form. **Thompson:** her koldan sonsaldan k örnek çek, ortala, en büyüğünü oyna (pratikte k = 1). Rastgelelik keşfi sağlar; $O(\log n)$ ve pratikte genelde en iyisi.
- **Alt sınır** (Lai–Robbins): her tutarlı algoritma için $\liminf\mathrm{Loss}_n/\ln n\ge\sum_{a:\Delta_a>0}\Delta_a/\mathrm{KL}(R(a)\,\|\,r^*)$.
