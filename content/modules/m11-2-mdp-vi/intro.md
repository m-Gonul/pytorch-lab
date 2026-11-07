**MDP:** durumlar $S=\{S_0,…,S_M\}$, eylemler $A$, geçiş modeli $P(s'|s,a)$, ödül $R(s,a)$, indirim $0\le\gamma\le1$, ufuk $h$. **Politika** $\pi(s)=a$ (tam gözlenebilir durum). Varsayımlar: tam gözlem, bilinen ve durağan model, ayrık zaman/durum/eylem. Amaç: beklenen indirimli ödül toplamını $\sum_t\gamma^tR(s_t,a_t)$ en büyük yapan politika. İndirim hem sonsuz toplamı sonlu kılar hem de erken ödülleri tercih eder.

**Değer fonksiyonu (sonlu ufuk):** $V_h(s)=R_h(s)$ (son adım ödülü genelde farklıdır: kazandı/kaybetti, elde kalan stok…) ve geriye doğru

$$V_t(s)=\max_a\Big[R(s,a)+\gamma\sum_{s'}P(s'|s,a)V_{t+1}(s')\Big].$$

Bu **değer yinelemesi** (dinamik programlama) her adım için ayrı bir eylem seçer: en iyi politika **durağan değildir** (kalan süreye bağlı).

**Sonsuz ufuk:** vektör biçiminde $V_{n}=\max_a(R^a+\gamma T^aV_{n-1})$ ($\max$ eleman bazında, $T^a$ satır-stokastik). Sabit nokta $V^*=H^*V^*$; açgözlü politika $\pi^*(s)=\arg\max_a[R(s,a)+\gamma\sum P V^*]$ **durağandır**.

**Politika değerlendirme:** sabit $\pi$ için $V=R_\pi+\gamma T_\pi V$ doğrusal: $V^\pi=(I-\gamma T_\pi)^{-1}R_\pi$ (ya da yinele).

**Yakınsama:** $H(V)=R+\gamma TV$ ve $H^*(V)=\max_a(R^a+\gamma T^aV)$ sonsuz normda **γ-büzülmedir**. Banach: tek sabit nokta, her başlangıçtan yakınsama, $\lVert H^{(n)}V-V^*\rVert_\infty\le\gamma^n\lVert V-V^*\rVert_\infty$. Pratik **durma sınırı**: $\lVert V_n-V_{n-1}\rVert_\infty\le\varepsilon$ ise $\lVert V^*-V_n\rVert_\infty\le\frac{\gamma}{1-\gamma}\varepsilon$.
