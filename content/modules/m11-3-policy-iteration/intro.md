**Fikir:** değer yinelemesi önce değeri yakınsatır, politikayı en sonda çıkarır. Politikayı doğrudan iyileştirebiliriz:

1. **Değerlendir:** $V^{\pi_n}=(I-\gamma T_{\pi_n})^{-1}R_{\pi_n}$ (ya da yinelemeyle).
2. **İyileştir:** $\pi_{n+1}(s)=\arg\max_a\big[R(s,a)+\gamma\sum_{s'}P(s'|s,a)V^{\pi_n}(s')\big]$.
3. Politika değişmeyene dek tekrarla.

**Lemma (politika iyileştirme):** $V^{\pi_{n+1}}\ge V^{\pi_n}$ (eleman bazında). İspat fikri: $H_{\pi_{n+1}}V^{\pi_n}=H^*V^{\pi_n}\ge H_{\pi_n}V^{\pi_n}=V^{\pi_n}$; $H_{\pi_{n+1}}$ monoton olduğu için tekrar uygulandıkça artar ve sabit noktası $V^{\pi_{n+1}}$'e yakınsar. **Teorem:** PI sonlu adımda $\pi^*$, $V^*$'a ulaşır (politika sayısı sonlu, döngü yok).

**Karşılaştırma:** VI ucuz iterasyonlar, çok iterasyon; PI pahalı iterasyonlar ($O(S^3)$ çözüm), çok az iterasyon (genelde < 10).

**Değiştirilmiş PI (MPI):** değerlendirmeyi kısalt: tam çözüm yerine $V\leftarrow H_{\pi}^{(k)}V$ ($k$ tarama), sonra iyileştir. $k=1$ → VI, $k=\infty$ → PI. Uygun $k$ ikisinden de hızlı olabilir. Durma: $\lVert V_n-V_{n-1}\rVert_\infty\le\varepsilon$ ise aynı türden sınır geçerlidir.
