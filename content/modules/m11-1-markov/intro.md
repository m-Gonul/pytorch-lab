**Rastgele değişken:** bir deneyin her sonucuna $\zeta$ bir sayı atayan kural, $\mathbf x(\zeta)$. **Rastgele süreç:** her sonuca bir **zaman fonksiyonu** atayan kural $\mathbf x(t,\zeta)$. Sabit $\zeta$ → bir örnek yol (gerçekleşme); sabit $t$ → bir rastgele değişken. Borsa fiyatları, günlük sıcaklık, gürültülü ölçümler…

**İstatistikler:** birinci derece dağılım $F(x;t)=P\{\mathbf x(t)\le x\}$, ikinci derece $F(x_1,x_2;t_1,t_2)$; ortalama $\eta(t)=E\{\mathbf x(t)\}$, otokorelasyon $R(t_1,t_2)=E\{\mathbf x(t_1)\mathbf x(t_2)\}$, otokovaryans $C(t_1,t_2)=R(t_1,t_2)-\eta(t_1)\eta(t_2)$.

**Durağanlık:** *katı* — tüm sonlu boyutlu dağılımlar zaman kaymasına değişmez. *Geniş anlamda (WSS)* — ortalama sabit, otokovaryans yalnız $\tau=t_1-t_2$'ye bağlı, ikinci moment sonlu. Gerçek sistemler nadiren katı durağandır; WSS çoğu zaman yeter. **Ergodiklik:** zaman ortalaması topluluk ortalamasına eşit (tek uzun gerçekleşmeden istatistik kestirilebilir).

**Wiener süreci** $W(t)$: $W(0)=0$, bağımsız artışlar, $W(t)-W(s)\sim\mathcal N(0,t-s)$, sürekli yollar. Rastgele yürüyüşün sürekli limiti.

**RL bağlantısı:** gözle → eylem seç → ödülü gör → tekrarla döngüsü, belirsizlikler yüzünden bir rastgele süreçtir. İlk model yalnız durum geçişleri: $s_0\to s_1\to s_2\to\cdots$.

**Markov süreçleri:** geçmişin tamamına bağlı $P(s_t|s_{t-1},…,s_0)$ modellemek imkansız; **k. derece Markov**: yalnız son $k$ durum; **birinci derece**: $P(s_t|s_{t-1})$ ("gelecek, şimdi verildiğinde geçmişten bağımsız"). **Durağan (zamanla homojen)**: geçiş olasılıkları $t$'ye bağlı değil. Durum kümesi sonlu ya da sayılabilir ise **Markov zinciri** — geçişler matris–vektör çarpımı:

$$T_{ij}=P(s_{t+1}=S_i\mid s_t=S_j),\qquad\mathbf p_{t+1}=T\mathbf p_t,\qquad\mathbf p_n=T^n\mathbf p_0.$$

(Bu modülde sütunlar "nereden", satırlar "nereye"; sütun toplamları 1. MDP modüllerinde satır-stokastik $P=T^\top$ da kullanılacak.) **Durağan dağılım:** $T\boldsymbol\pi=\boldsymbol\pi$ — özdeğer 1'in özvektörü; indirgenemez ve periyodik olmayan zincirde tek ve her başlangıçtan yakınsanır.

:::sezgi Pekiştirmeli öğrenme nerede duruyor?
**Denetimli öğrenme:** her girdi için doğru cevap (etiket) verilir; hata doğrudan ölçülür. **Denetimsiz:** etiket yok; verinin yapısı (küme, yoğunluk, düşük boyutlu temsil) aranır. **Yarı denetimli:** az etiketli + çok etiketsiz veri. **Pekiştirmeli öğrenme:** doğru cevap yok, yalnız bir **ödül sinyali** var — üstelik gecikmeli (iyi bir hamlenin değeri oyunun sonunda belli olur) ve ajanın kendi eylemleri hangi veriyi göreceğini belirler (veri i.i.d. değil). Kökü hayvan psikolojisinde: Thorndike'ın etki yasası, Pavlov ve Skinner'ın koşullanması, ödül tahmin hatasının dopamin nöronlarıyla ilişkisi (TD hatası, M11.4). Uygulamalar: oyunlar (Atari, Go — M11.6, M11.11), robotik, envanter ve dinamik fiyatlandırma (M11.2), öneri sistemleri ve reklam (çevrimiçi karar: haydutlar, M11.7–11.8), algoritmik alım satım (durağan olmayan, gürültülü ödül — dikkatli değerlendirme gerektirir), büyük dil modellerinin insan tercihleriyle hizalanması (RLHF, M4.16). Bu modülde temel taş: geleceğin yalnız şimdiye bağlı olduğu rastgele süreçler — Markov zincirleri.
:::
