**Model olmadan kontrol:** $V^\pi$'den açgözlü politika çıkarmak için $P$ gerekir ($\arg\max_a R+\gamma\sum P V$). $Q^\pi(s,a)$ öğrenirsek $\pi'(s)=\arg\max_aQ(s,a)$ modelsiz olur. Bellman optimallik: $Q^*(s,a)=\mathbb E[r]+\gamma\sum_{s'}P(s'|s,a)\max_{a'}Q^*(s',a')$, $V^*=\max_aQ^*$, $\pi^*=\arg\max_aQ^*$.

**Monte Carlo kontrol:** politika değerlendirme (getirileri ortala) ile iyileştirmeyi dönüşümlü yap — politika yinelemesinin örneklemeli hâli. Her (s,a)'nın denenmesi için ya **keşifçi başlangıçlar** (rastgele (s,a)'dan başla) ya da ε-yumuşak politikalar.

**TD kontrol:**

| yöntem | hedef | tür |
|---|---|---|
| SARSA | $r+\gamma Q(s',a')$, $a'\sim\pi$ | politika üstü |
| beklenen SARSA | $r+\gamma\sum_{a'}\pi(a'|s')Q(s',a')$ | ikisi de olabilir |
| Q-öğrenme | $r+\gamma\max_{a'}Q(s',a')$ | politika dışı |

Q-öğrenme (Watkins 1989): `eylem seç ve uygula → r, s' gözle → n(s,a)++ → α = 1/n(s,a) → Q(s,a) += α(r + γ max Q(s',·) − Q(s,a)) → s ← s'`.

**Keşif mi sömürü mü:** hep açgözlü seçmek hiç denenmemiş eylemleri asla öğrenmez; hep rastgele seçmek öğrendiğini kullanmaz. ε-açgözlü (ε olasılıkla rastgele), Boltzmann ($\pi(a)\propto e^{Q(s,a)/\tau}$), iyimser başlangıç ($Q_0$ büyük: denenmemiş eylem cazip). GLIE: her çift sonsuz kez, politika sınırda açgözlü.

**Maksimizasyon yanlılığı:** gürültülü kestirimlerin maksimumu, maksimumun yanlı kestirimidir. Çift Q-öğrenme: iki tablo; $Q_1$ ile seç, $Q_2$ ile değerlendir: $Q_1(s,a)\leftarrow Q_1+\alpha[r+\gamma Q_2(s',\arg\max_{a'}Q_1(s',a'))-Q_1]$ (yazı-turaya göre roller değişir).
