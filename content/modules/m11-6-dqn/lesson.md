### Tablo sığmayınca {#buyuk}

M11.4–M11.5'teki yöntemler $V(s)$ ya da $Q(s,a)$'yı bir tabloda tutuyordu. Her $(s,a)$ çifti için bir hücre vardı ve her hücre ayrı ayrı öğreniliyordu. Pratik problemlerin neredeyse hepsinde bu imkânsızdır:

- **Go:** 19 × 19 tahtada her nokta boş, siyah ya da beyaz olabilir: $3^{361}\approx1.7\cdot10^{172}$ dizilim. Bunların yaklaşık $2.1\cdot10^{170}$'i kurallara uygundur. (Slayttaki "3331" bir dizgi hatasıdır; doğrusu $3^{361}$'dir.)
- **Atari:** durum ekran görüntüsüdür, 210 × 160 × 3 piksel. Olası ekranların sayısı astronomiktir ve aynı ekranın tam olarak tekrar görülmesi neredeyse hiç olmaz.
- **Sürekli durumlar:** konum, hız, açı gibi büyüklükler ızgaraya bölünür. Boyut sayısıyla üstel olarak artan sayıda hücre çıkar.

Tablo hem bellek hem veri sorunudur. Her hücrenin değerinin öğrenilmesi için o hücrenin defalarca ziyaret edilmesi gerekir. Çözüm **genellemedir**: benzer durumların değerleri benzer olmalıdır. Öğrenilmesi gereken üç fonksiyon vardır: değer $V(s)\in\mathbb R$, politika $\pi:s\to a$ ve Q-fonksiyonu $Q(s,a)\in\mathbb R$. Tablo kullanılamıyorsa bunlar parametreli fonksiyonlarla yaklaşılır.

### Q-fonksiyonunu yaklaşmak {#yaklasim}

Durum bir öznitelik vektörüyle temsil edilir: $s=(x_1,\dots,x_N)^\top$. Atari'de öznitelikler ekran görüntülerinden bir evrişimli ağla çıkarılır. Q-fonksiyonu iki biçimde yaklaşılabilir:

- **Doğrusal:** $Q(s,a)\approx\sum_iw_{a,i}x_i$. Her eylem için ayrı bir ağırlık vektörü vardır.
- **Doğrusal olmayan:** $Q(s,a)\approx q(x_1,\dots,x_N;w_1,\dots,w_K)$, ör. bir sinir ağı.

Ağırlıklar, kestirim ile hedef arasındaki kare hatayı küçültecek biçimde öğrenilir. Q-öğrenmenin hedefi $r+\gamma\max_{a^+}Q_w(s^+,a^+)$'dır. Ama burada bir sorun vardır: $w$ hem kestirimde hem hedefte görünür. $w$ değişince hem kestirim hem de ulaşılmak istenen hedef değişir. Çözüm, hedefteki ağırlıkları "dondurulmuş" bir kopya $\bar w$ ile değiştirmektir:
$$\text{Hata}(w)=\tfrac12\Bigl[Q_w(s,a)-r-\gamma\max_{a^+}Q_{\bar w}(s^+,a^+)\Bigr]^2,$$
$$\frac{\partial\,\text{Hata}}{\partial w}=\Bigl[Q_w(s,a)-r-\gamma\max_{a^+}Q_{\bar w}(s^+,a^+)\Bigr]\frac{\partial Q_w(s,a)}{\partial w}.$$
Güncelleme $w\leftarrow w-\alpha\,\partial\text{Hata}/\partial w$'dir.

:::türetim Yarı-gradyan: neden "gradyan Q-öğrenme" bir gradyan değildir
Hedefte $\bar w=w$ alınırsa (dondurmadan), yukarıdaki ifade gerçek kaybın gradyanı değildir. $\tfrac12[Q_w(s,a)-r-\gamma\max Q_w(s^+,\cdot)]^2$'nin tam gradyanı hedefin türevini de içerirdi:
$$\delta\Bigl[\nabla_wQ_w(s,a)-\gamma\nabla_w\max_{a^+}Q_w(s^+,a^+)\Bigr].$$
Q-öğrenme ve TD ikinci terimi atar. Bu yüzden **yarı-gradyan** yöntemlerdir. Hedefi sabit bir sayı gibi görüp yalnız kestirimi ona doğru iterler. Bu seçim bilinçlidir: tam gradyan ("artık gradyan", Baird 1995) kararlıdır ama stokastik ortamlarda yanlış değere yakınsar ve yavaştır. Bedeli şudur: yarı-gradyan güncelleme herhangi bir fonksiyonun inişi değildir ve yakınsama güvencesi gradyan inişinden gelmez.
:::

### Yakınsama: tablo, doğrusal ve doğrusal olmayan {#yakinsama}

**Tablo Q-öğrenme**, M11.5'teki koşullarla $Q^*$'a yakınsar: her $(s,a)$ sonsuz kez ziyaret edilir ve $\alpha_{n(s,a)}$ Robbins–Monro koşullarını sağlar. Tablo, her $(s,a)$ için tek-sıcak bir öznitelik vektörüyle doğrusal yaklaşımın özel bir hâlidir.

:::uyarı Kaynak düzeltmesi (lect_7): doğrusal Q-öğrenme her zaman yakınsamaz
Slayt, doğrusal gradyan Q-öğrenmenin "aynı koşullar altında en iyi Q-fonksiyonuna yakınsadığını" ve kestirimde ve hedefte aynı $w$ kullanılsa da "sorunsuz" yakınsadığını söylüyor. Bu yanlıştır:

1. **Doğrusal yaklaşımda bile ıraksama olur.** Baird'in (1995) karşı örneğinde ödüller hep 0'dır ve doğru çözüm ($w=0$) temsil edilebilir. Yine de politika dışı yarı-gradyan TD'nin ağırlıkları sonsuza gider (aşağıdaki deney). DQN makalesi de bunu not eder: Q-öğrenme ile doğrusal olmayan yaklaşımın ya da politika dışı öğrenmenin birleşimi Q-ağını ıraksatabilir.
2. **Yakınsasa bile hedef $Q^*$ değildir.** Doğrusal politika üstü TD(0) yakınsar (Tsitsiklis ve Van Roy 1997). Ama vardığı nokta, izdüşürülmüş Bellman denkleminin çözümüdür: özniteliklerin gerilediği uzaydaki en iyi temsil, ve o bile tam değil. $Q^*$ öznitelik uzayında temsil edilemiyorsa ona ulaşılamaz.
3. Doğrusal Q-öğrenmenin yakınsaması ancak kısıtlayıcı koşullarda kanıtlanmıştır (ör. Melo ve ark. 2008); genel bir güvence yoktur.
:::

**Ölümcül üçlü.** Sutton ve Barto ıraksamanın üç bileşenin bir arada olmasından kaynaklandığını vurgular:

1. Fonksiyon yaklaşımı (genelleme);
2. Bootstrapping (hedefte kendi tahminini kullanma);
3. Politika dışı öğrenme (davranış dağılımı ile hedef politikanın dağılımının farklı olması).

Herhangi ikisi tek başına güvenlidir. Tablo + bootstrapping + politika dışı, Q-öğrenmedir. Yaklaşım + bootstrapping + politika üstü, doğrusal TD'dir. Yaklaşım + politika dışı + MC, önem örneklemeli regresyondur. Üçü birden olunca ağırlıklar ıraksayabilir. Gradyan-TD yöntemleri (GTD2, TDC) izdüşürülmüş Bellman hatasının **gerçek** gradyanını izleyerek üçlüde bile ıraksamaz.

::lab l1

**Doğrusal olmayan yaklaşımda** sorun daha da büyüktür. Aynı $w$ kestirimde ve hedefte kullanılırsa, Robbins–Monro koşulları sağlansa bile ıraksama görülebilir. Bu şaşırtıcı değildir: $w$ üzerindeki her gradyan adımı, optimize edilmeye çalışılan hedefi de değiştirir. Pratikte iki hile kullanılır: deneyim tekrarı ve hedef ağ.

### Deneyim tekrarı {#tekrar}

Her geçiş $(s,a,r,s^+)$ bir tampona yazılır. Her adımda tampondan rastgele bir mini yığın çekilir ve Q-öğrenme güncellemesi bu yığın üzerinde yapılır. DQN makalesi (Mnih ve ark. 2013) üç yarar sayar:

1. **Veri verimliliği.** Her deneyim birçok ağırlık güncellemesinde kullanılır.
2. **İlişkinin kırılması.** Ardışık örnekler güçlü biçimde ilişkilidir; onlardan doğrudan öğrenmek verimsizdir. Rastgele örnekleme ilişkiyi kırar ve güncellemelerin varyansını azaltır.
3. **Geri besleme döngülerinin önlenmesi.** Politika üstü öğrenmede mevcut parametreler bir sonraki eğitim verisini belirler. En iyi eylem "sola git" ise veri sol taraftan gelen örneklerle dolar. En iyi eylem sağa dönünce eğitim dağılımı da sağa kayar. Böyle döngüler parametreleri kötü bir yerel minimumda tutabilir ya da ıraksatabilir. Tekrarla davranış dağılımı birçok önceki durumun ortalaması olur ve salınım yumuşar.

Tekrar **politika dışı** öğrenmeyi zorunlu kılar. Tampondaki geçişler eski parametrelerin politikasıyla toplanmıştır. SARSA gibi politika üstü bir hedef yanlış olur. Q-öğrenmenin seçilmesinin bir nedeni budur. DQN tamponda son 1 milyon kareyi tutar ve 32'lik mini yığınlarla düzgün örnekler. Önemli geçişleri daha sık örnekleyen **öncelikli tekrar** (Schaul ve ark. 2016) sonraki bir iyileştirmedir.

Aşağıdaki deney ikinci yararı gözetimli bir benzerle gösteriyor. Durum yavaşça ileri geri kayıyor ve bir fonksiyon bu ilişkili örneklerden öğreniliyor. Küresel özniteliklerde çevrimiçi SGD o an bulunduğu bölgeye uyup uzak bölgeleri bozuyor. Tekrar aynı sayıda güncellemeyle hatayı on kattan fazla düşürüyor.

::lab l2

### Hedef ağ {#hedef-ag}

Gradyan Q-öğrenmedeki dondurulmuş $\bar w$ ayrı bir ağdır, **hedef ağ**:
$$y=r+\gamma\max_{a^+}Q_{\bar w}(s^+,a^+).$$
Q-ağı ($w$) her adımda güncellenir. Hedef ağ her $C$ adımda bir $w$'nin kopyasıyla değiştirilir ($\bar w\leftarrow w$). Arada hedef sabittir ve problem $C$ adım boyunca sıradan bir denetimli regresyona benzer. Hareketli hedefin yarattığı geri besleme kırılır. Nature sürümünde $C=10\,000$ güncellemedir. Yumuşak güncelleme $\bar w\leftarrow\tau w+(1-\tau)\bar w$ (DDPG, $\tau\approx0.005$) aynı fikrin sürekli bir biçimidir.

:::uyarı Kaynak düzeltmesi (lect_7): DQN'nin iki sürümü
Slaytlar DQN'yi "derin ağ + deneyim tekrarı + hedef ağ" olarak tanımlıyor ve arXiv 2013 makalesine (numarası slaytta "1312.05602" yazılmış; doğrusu 1312.5602) bağlıyor. 2013 makalesinde ayrı bir hedef ağ **yoktur**. Algoritma 1'deki hedef, güncellenmekte olan aynı $\theta$ ile hesaplanır. Kuramsal türetimde $\theta_{i-1}$ "sabit tutulur", ama uygulamada ayrı bir kopya yoktur. Hedef ağ, 2015 Nature makalesinde ($C=10\,000$) eklenir. Aynı makalede ağ büyütülür (3 evrişim katmanı, 512 birim), 49 oyunda değerlendirme yapılır ve hata terimi $[-1,1]$'e kırpılır (Huber kaybına denk).
:::

### DQN: Atari'de insan düzeyi {#dqn}

Mnih ve ark. (2013), "derin Q-ağı" (DQN) ile aynı ağ, algoritma ve hiper parametrelerle yedi Atari oyununu yalnız ham piksellerden öğrendi. Ayrıntılar şöyledir:

- **Ön işleme** ($\phi$): 210 × 160 RGB kare gri tona çevrilir, 110 × 84'e küçültülür, oyun alanını kapsayan 84 × 84'lük bir bölge kırpılır. Son 4 kare üst üste konur. Tek bir kare hareket yönünü göstermez, dört kare gösterir; durum bu yolla Markov'a yaklaştırılır (M11.1).
- **Ağ:** girdi 84 × 84 × 4'tür. Ardından 16 tane 8 × 8 filtre, adım 4; 32 tane 4 × 4 filtre, adım 2; 256 birimli tam bağlı katman gelir. Çıktıda her eylem için bir değer vardır. Bütün eylemlerin Q değerleri tek ileri geçişte hesaplanır; $\max_a$ için $|\mathcal A|$ geçiş gerekmez (Şekil 1).
- **Eğitim:** RMSProp, mini yığın 32. ε ilk 1 milyon karede 1'den 0.1'e doğrusal azaltılır, sonra 0.1'de sabit kalır. Toplam 10 milyon kare eğitilir; tampon son 1 milyon kareyi tutar.
- **Kare atlama:** ajan her $k$. karede eylem seçer, arada son eylem tekrarlanır. $k=4$'tür. Space Invaders'da lazerler yanıp söndüğü için $k=3$ kullanılır; oyunlar arasındaki tek ayar farkı budur.
- **Ödül kırpma:** puan ölçekleri oyundan oyuna çok farklıdır. Bu yüzden eğitimde pozitif ödüller +1, negatifler −1 yapılır. Aynı öğrenme oranı her oyunda çalışır. Bedeli, ajanın büyük ve küçük ödülü ayıramamasıdır.

<figure class="fig wide"><svg viewBox="0 0 640 262" role="img" aria-label="Atari için DQN ağlarının katmanları ve boyutları, 2013 ve 2015 sürümleri"><defs><marker id="ah6" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs><text x="14" y="22" class="sm">2013 (arXiv): 2 evrişim + 1 tam bağlı katman; düzleştirilmiş vektör 2592; toplam 680 770 parametre</text><rect x="14" y="34" width="55.440000000000005" height="55.440000000000005" class="soft-a" stroke="currentColor" stroke-width="0.8"/><rect x="18" y="30" width="55.440000000000005" height="55.440000000000005" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="14" y="104" class="sm">84×84×4</text><text x="14" y="120" class="sm" font-size="10">girdi</text><path d="M77.44,60 L114,60" class="mute" marker-end="url(#ah6)"/><rect x="114.0" y="34" width="13.200000000000001" height="13.200000000000001" class="soft-a" stroke="currentColor" stroke-width="0.8"/><rect x="118.0" y="30" width="13.200000000000001" height="13.200000000000001" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="114.0" y="104" class="sm">20×20×16</text><text x="114.0" y="120" class="sm" font-size="10">conv 16@8×8/4</text><path d="M135.2,60 L214.0,60" class="mute" marker-end="url(#ah6)"/><rect x="214.0" y="34" width="10" height="10" class="soft-a" stroke="currentColor" stroke-width="0.8"/><rect x="218.0" y="30" width="10" height="10" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="214.0" y="104" class="sm">9×9×32</text><text x="214.0" y="120" class="sm" font-size="10">conv 32@4×4/2</text><path d="M232.0,60 L314.0,60" class="mute" marker-end="url(#ah6)"/><rect x="314.0" y="30" width="14" height="60" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="314.0" y="104" class="sm">256</text><text x="314.0" y="120" class="sm" font-size="10">tam bağlı 256</text><path d="M332.0,60 L404.0,60" class="mute" marker-end="url(#ah6)"/><rect x="404.0" y="30" width="14" height="60" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="404.0" y="104" class="sm">18</text><text x="404.0" y="120" class="sm" font-size="10">çıkış |A| = 18</text><text x="14" y="144" class="sm">2015 (Nature): 3 evrişim + 1 tam bağlı katman; düzleştirilmiş vektör 3136; toplam 1 693 362 parametre</text><rect x="14" y="156" width="55.440000000000005" height="55.440000000000005" class="soft-a" stroke="currentColor" stroke-width="0.8"/><rect x="18" y="152" width="55.440000000000005" height="55.440000000000005" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="14" y="226" class="sm">84×84×4</text><text x="14" y="242" class="sm" font-size="10">girdi</text><path d="M77.44,182 L114,182" class="mute" marker-end="url(#ah6)"/><rect x="114.0" y="156" width="13.200000000000001" height="13.200000000000001" class="soft-a" stroke="currentColor" stroke-width="0.8"/><rect x="118.0" y="152" width="13.200000000000001" height="13.200000000000001" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="114.0" y="226" class="sm">20×20×32</text><text x="114.0" y="242" class="sm" font-size="10">conv 32@8×8/4</text><path d="M135.2,182 L214.0,182" class="mute" marker-end="url(#ah6)"/><rect x="214.0" y="156" width="10" height="10" class="soft-a" stroke="currentColor" stroke-width="0.8"/><rect x="218.0" y="152" width="10" height="10" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="214.0" y="226" class="sm">9×9×64</text><text x="214.0" y="242" class="sm" font-size="10">conv 64@4×4/2</text><path d="M232.0,182 L314.0,182" class="mute" marker-end="url(#ah6)"/><rect x="314.0" y="156" width="10" height="10" class="soft-a" stroke="currentColor" stroke-width="0.8"/><rect x="318.0" y="152" width="10" height="10" class="soft-a" stroke="currentColor" stroke-width="0.8"/><text x="314.0" y="226" class="sm">7×7×64</text><text x="314.0" y="242" class="sm" font-size="10">conv 64@3×3/1</text><path d="M332.0,182 L414.0,182" class="mute" marker-end="url(#ah6)"/><rect x="414.0" y="152" width="14" height="60" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="414.0" y="226" class="sm">512</text><text x="414.0" y="242" class="sm" font-size="10">tam bağlı 512</text><path d="M432.0,182 L504.0,182" class="mute" marker-end="url(#ah6)"/><rect x="504.0" y="152" width="14" height="60" class="soft-b" stroke="currentColor" stroke-width="0.8"/><text x="504.0" y="226" class="sm">18</text><text x="504.0" y="242" class="sm" font-size="10">çıkış |A| = 18</text></svg><figcaption>Şekil 1 — DQN ağı: son dört gri tonlu 84 × 84 kare girdi, her eylem için bir Q değeri çıktı (tek ileri geçişte bütün eylemler). Boyutlar ve parametre sayıları kenar = ⌊(H − k)/s⌋ + 1 ile hesaplandı; |A| = 18 (Atari'nin en geniş eylem kümesi).</figcaption></figure>

Sonuçlar ($\epsilon=0.05$ ile ortalama puan, makalenin Tablo 1'i):

| | Beam Rider | Breakout | Enduro | Pong | Q*bert | Seaquest | Space Invaders |
|---|---|---|---|---|---|---|---|
| rastgele | 354 | 1.2 | 0 | −20.4 | 157 | 110 | 179 |
| Sarsa (doğrusal) | 996 | 5.2 | 129 | −19 | 614 | 665 | 271 |
| DQN | 4092 | 168 | 470 | 20 | 1952 | 1705 | 581 |
| insan | 7456 | 31 | 368 | −3 | 18900 | 28010 | 3690 |

DQN yedi oyunun altısında önceki bütün yöntemleri geçti ve üçünde (Breakout, Enduro, Pong) uzman bir insan oyuncuyu aştı. En iyi doğrusal yöntemle (öznitelikleri elle tasarlanmış Sarsa) arasındaki fark Pong dışında iki ile otuz kat arasındadır (Breakout'ta 5.2'ye karşı 168). Q*bert ve Seaquest gibi uzun vadeli strateji isteyen oyunlarda insan hâlâ çok öndeydi. 2015 sürümü 49 oyunda değerlendirildi ve 29'unda profesyonel bir insan test oyuncusunun puanının %75'ini aştı.

DQN 2015'te en ileri yöntemdi. Sonraki yıllarda birçok iyileştirme eklendi:

- Double DQN (2016): hedefte eylemi $w$ ile seçip $\bar w$ ile değerlendirerek maksimizasyon yanlılığını (M11.5) azaltır: $y=r+\gamma Q_{\bar w}(s^+,\arg\max_{a^+}Q_w(s^+,a^+))$.
- Öncelikli tekrar (2016).
- Dueling ağlar (2016): $Q=V+A$ ayrıştırması.
- Dağılımsal RL (C51, 2017): getirinin beklentisi yerine dağılımını öğrenir.
- Rainbow (2018): bunların hepsinin birleşimi.
- Agent57 (2020): 57 oyunun hepsinde insan düzeyi.

Model tabanlı MuZero (M11.11) ve politika gradyanı yöntemleri (M11.9, M11.12) de bu alanda güçlüdür.

### Özet {#ozet}

- Durum uzayı büyükse (Go'da $3^{361}$ dizilim, Atari'de ham pikseller) tablo yerine parametreli bir fonksiyon kullanılır; genelleme hem belleği hem veri ihtiyacını çözer.
- Gradyan Q-öğrenme, $\tfrac12[Q_w(s,a)-r-\gamma\max Q_{\bar w}(s^+,\cdot)]^2$'yi küçültür. Hedefin türevi alınmaz: yöntem yarı-gradyandır.
- Tablo Q-öğrenme yakınsar. Doğrusal Q-öğrenme her zaman yakınsamaz (Baird); politika üstü doğrusal TD yakınsar, ama izdüşürülmüş sabit noktaya. Ölümcül üçlü: yaklaşım + bootstrapping + politika dışı.
- Deneyim tekrarı veriyi yeniden kullanır, ardışık örneklerin ilişkisini kırar ve geri besleme döngülerini yumuşatır; politika dışı öğrenme gerektirir.
- Hedef ağ, hedefi C adım dondurarak hareketli hedef sorununu hafifletir. Ayrı hedef ağ 2015 Nature sürümünde gelir; 2013 makalesinde yoktur.
- DQN: 4 karelik 84 × 84 girdi, evrişimli ağ, her eylem için bir çıktı, ε azaltma, kare atlama, ödül kırpma. 2013'te yedi oyunun altısında en iyi, üçünde insan üstü.
