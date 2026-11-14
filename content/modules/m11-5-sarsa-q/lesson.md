### Değerlendirmeden kontrole: neden Q? {#q-deger}

M11.4 sabit bir politikanın değerini model olmadan kestirdi. Asıl hedef ise en iyi politikayı bulmaktır (kontrol problemi). Politika yinelemesinin kalıbı (değerlendir, sonra açgözlü iyileştir) burada doğrudan işlemez. $V^\pi$'den açgözlü politika çıkarmak için
$$\pi'(s)=\arg\max_a\Bigl[R(s,a)+\gamma\sum_{s^+}P(s^+\mid s,a)V^\pi(s^+)\Bigr]$$
hesaplanmalıdır. Bunun için yine model gerekir. Çözüm, durum değeri yerine **durum–eylem değerini** öğrenmektir:
$$Q^\pi(s,a)=E\bigl[r\mid s,a\bigr]+\gamma\sum_{s^+}P(s^+\mid s,a)\,V^\pi(s^+).$$
$Q^\pi(s,a)$, $s$'de $a$'yı uygulayıp sonra $\pi$'yi izlemenin beklenen getirisidir. Açgözlü politika modelsiz çıkar: $\pi'(s)=\arg\max_aQ^\pi(s,a)$. Bedeli tablonun $|\mathcal A|$ kat büyümesidir.

:::uyarı Kaynak düzeltmesi (lect_6): $Q^\pi$'nin tanımı
Slaytta $Q^\pi(s,a)=E[r\mid s,a]+\gamma\sum_{s^+}P(s^+\mid s,\pi(s))V^\pi(s^+)$ yazıyor. Geçiş olasılığı $\pi(s)$ ile değil, **$a$** ile koşullanmalıdır: ilk adımda uygulanan eylem $a$'dır, $\pi$ ancak ondan sonra devreye girer. Yazıldığı biçimde ödül $a$'ya, geçiş $\pi(s)$'ye bağlıdır. Bu tutarsızdır ve $\arg\max_aQ$ ile politika iyileştirmeyi bozar, çünkü eylemlerin gelecekteki etkisi kaybolur.
:::

**Bellman denklemleri.** En iyi değerler için iki biçim vardır:
$$V^*(s)=\max_a\Bigl[E[r\mid s,a]+\gamma\sum_{s^+}P(s^+\mid s,a)V^*(s^+)\Bigr],$$
$$Q^*(s,a)=E[r\mid s,a]+\gamma\sum_{s^+}P(s^+\mid s,a)\max_{a^+}Q^*(s^+,a^+).$$
İkisi arasında $V^*(s)=\max_aQ^*(s,a)$ ve $\pi^*(s)=\arg\max_aQ^*(s,a)$ bağı vardır. $Q^*$'ın denkleminde maksimum beklentinin **içindedir**. Bu yüzden tek bir örnekle kestirilebilir ve Q-öğrenme bundan doğar.

### Monte Carlo kontrol {#mc-kontrol}

MC kontrolün hedefi tek bir yörüngenin getirisidir. $s$'de $a$ seçilir, ilk ödül $r_0$ alınır, gerisi $\pi$ ile oynanır:
$$G^a=r_0+\sum_{t\ge1}\gamma^tr_t.$$
Politika yinelemesindeki gibi iki adım dönüşümlü uygulanır:

- **Değerlendirme:** $Q(s,a)\leftarrow Q(s,a)+\alpha_{n(s,a)}\bigl[G^a-Q(s,a)\bigr]$.
- **İyileştirme:** $\pi'(s)=\arg\max_aQ(s,a)$.

Sayaç durum–eylem çifti başına tutulur: $n(s,a)$. (Slaytlarda öğrenme oranının indeksi $n(s)$ diye yazılmış; güncellenen öğe $Q(s,a)$ olduğu için sayaç $n(s,a)$ olmalıdır. Q-öğrenme sözde kodu ise sayacı doğru biçimde $n(s,a)$ olarak artırıyor ama güncellemede yine $\alpha_{n(s)}$ yazıyor.) Bir sorun kalır. Açgözlü politika bazı eylemleri hiç seçmezse onların $Q$ değerleri hiç kestirilmez. Çözümlerden biri **keşifçi başlangıçlardır**: her bölüm rastgele bir $(s,a)$ çiftinden başlar. Daha genel çözüm aşağıdaki keşif kurallarıdır.

### Q-öğrenme ve SARSA {#q-ogrenme}

**Q-öğrenme** (Watkins 1989), $Q^*$'ın Bellman denklemindeki beklentiyi tek bir geçiş örneğiyle değiştirir. $s$'de $a$ uygulanır, $r$ ve $s^+$ gözlenir:
$$Q^*(s,a)\approx r+\gamma\max_{a^+}Q^*(s^+,a^+)$$
$$Q(s,a)\leftarrow Q(s,a)+\alpha\Bigl[r+\gamma\max_{a^+}Q(s^+,a^+)-Q(s,a)\Bigr].$$

```text
Q-öğrenme:
    her (s, a) için Q(s, a) ← 0,  n(s, a) ← 0
    s ← başlangıç
    tekrarla:
        a'yı keşif kuralıyla seç (ör. ε-açgözlü);  uygula;  r ve s⁺'yı gözle
        n(s, a) ← n(s, a) + 1;   α ← 1 / n(s, a)
        Q(s, a) ← Q(s, a) + α [ r + γ max_{a⁺} Q(s⁺, a⁺) − Q(s, a) ]
        s ← s⁺
    Q yakınsayana dek
```

Q-öğrenme **politika dışıdır** (off-policy). Davranış politikası (keşif yapan ε-açgözlü) ile öğrenilen politika (açgözlü, hedefteki $\max$) farklıdır. Hangi eylem seçilirse seçilsin hedef, en iyi eylemin değerini kullanır.

**SARSA** (Rummery ve Niranjan 1994) hedefte gerçekten seçilecek sonraki eylemi kullanır. Adını güncellemenin kullandığı beşliden alır: $(s,a,r,s^+,a^+)$.
$$Q(s,a)\leftarrow Q(s,a)+\alpha\bigl[r+\gamma Q(s^+,a^+)-Q(s,a)\bigr],\qquad a^+\sim\pi(\cdot\mid s^+).$$
SARSA **politika üstüdür** (on-policy). İzlenen politikanın, keşif dahil, değerini öğrenir. **Beklenen SARSA** rastgele $a^+$ yerine onun beklentisini alır: $r+\gamma\sum_{a^+}\pi(a^+\mid s^+)Q(s^+,a^+)$. Aynı beklentiyi daha düşük varyansla kestirir. Hedef politika açgözlü seçilirse beklenen SARSA Q-öğrenmeye dönüşür.

Fark **uçurum yürüyüşünde** belirginleşir (Sutton ve Barto Örnek 6.6). 4 × 12 bir ızgarada sol alttan sağ alta gidilir. Alt sıradaki ara hücreler uçurumdur: girmek −100 ödül verir ve başa döndürür. Diğer her adım −1'dir. En kısa yol (13 adım) uçurumun hemen üstünden geçer. Q-öğrenme bu yolu öğrenir. Ama ε-açgözlü davranırken kenarda her rastgele adım uçuruma düşürür ve eğitim boyunca toplanan ödül düşer. SARSA keşfi değere katar ve uçurumdan uzak, daha uzun ama güvenli bir yolu öğrenir. Eğitimde daha çok ödül toplar. $\epsilon\to0$ giderken iki yöntem aynı en iyi politikaya yakınsar.

::lab l1

### Keşif ve sömürü {#kesif}

Q-öğrenmenin ilk satırı, "eylemi seç", göründüğünden zordur. İki uç vardır:

- **Sömürü:** her zaman en yüksek $Q$'lu eylem seçilir. Ama öğrenilen $Q$ gerçek değer değildir; hiç denenmemiş $(s,a)$ çiftleri olabilir. Başta şans eseri iyi görünen bir eyleme takılıp daha iyisi hiç keşfedilmeyebilir. Sonuç en iyi olmayan bir politikadır.
- **Keşif:** rastgele (en iyi görünmeyen) eylemler denenir. Ama öğrenilenleri hiç kullanmayan bir ajan ödül de toplamaz.

Denge gerekir. Yaygın kurallar şunlardır:

- **ε-açgözlü:** $\epsilon$ olasılıkla rastgele bir eylem, $1-\epsilon$ olasılıkla açgözlü eylem seçilir. Açgözlü eylemin olasılığı $1-\epsilon+\epsilon/|\mathcal A|$'dır. $\epsilon$ genellikle zamanla azaltılır: 1.0'dan başlayıp doğrusal ya da üstel olarak küçük bir alt sınıra iner.
- **Boltzmann (softmax):** $P(a)=e^{Q(s,a)/T}/\sum_be^{Q(s,b)/T}$. $T$ sıcaklıktır. Yüksek $T$'de üsler küçülür ve dağılım düzgüne yaklaşır; düşük $T$'de en büyük değer baskın olur ve dağılım tek bir eyleme yoğunlaşır. ε-açgözlüden farkı, kötü eylemleri değerlerine göre ayırmasıdır (Şekil 2).
- **İyimser başlangıç:** $Q$ gerçekçi olmayan büyük bir değerle başlatılır. Her denenen eylem "hayal kırıklığı" yaratır ve ajan henüz denenmemiş eylemlere yönelir. Keşif kendiliğinden ve sistematik olur, ama yalnız başlangıçta işe yarar.
- **Belirsizlik bonusu (UCB):** az denenmiş eylemlere $c\sqrt{\ln t/n(s,a)}$ kadar bonus verilir. Haydutlarda (M11.7) ayrıntılı işleniyor.

<figure class="fig wide"><svg viewBox="0 0 640 256" role="img" aria-label="Epsilon açgözlü ve Boltzmann keşfinde üç eylemin seçilme olasılıkları"><text x="50" y="16" class="sm">ε-açgözlü</text><path d="M50,180 L290,180" class="mute"/><text x="44" y="184" text-anchor="end" class="sm">0</text><path d="M50,105.0 L290,105.0" class="mute"/><text x="44" y="109.0" text-anchor="end" class="sm">0.5</text><path d="M50,30 L290,30" class="mute"/><text x="44" y="34" text-anchor="end" class="sm">1</text><text x="50.0" y="196" text-anchor="middle" class="sm">0</text><text x="170.0" y="196" text-anchor="middle" class="sm">0.5</text><text x="290.0" y="196" text-anchor="middle" class="sm">1</text><text x="170.0" y="212" text-anchor="middle" class="sm">ε</text><path d="M50.0,180.0 L54.8,179.0 L59.6,178.0 L64.4,177.0 L69.2,176.0 L74.0,175.0 L78.8,174.0 L83.6,173.0 L88.4,172.0 L93.2,171.0 L98.0,170.0 L102.8,169.0 L107.6,168.0 L112.4,167.0 L117.2,166.0 L122.0,165.0 L126.8,164.0 L131.6,163.0 L136.4,162.0 L141.2,161.0 L146.0,160.0 L150.8,159.0 L155.6,158.0 L160.4,157.0 L165.2,156.0 L170.0,155.0 L174.8,154.0 L179.6,153.0 L184.4,152.0 L189.2,151.0 L194.0,150.0 L198.8,149.0 L203.6,148.0 L208.4,147.0 L213.2,146.0 L218.0,145.0 L222.8,144.0 L227.6,143.0 L232.4,142.0 L237.2,141.0 L242.0,140.0 L246.8,139.0 L251.6,138.0 L256.4,137.0 L261.2,136.0 L266.0,135.0 L270.8,134.0 L275.6,133.0 L280.4,132.0 L285.2,131.0 L290.0,130.0" class="cb w2"/><path d="M50.0,180.0 L54.8,179.0 L59.6,178.0 L64.4,177.0 L69.2,176.0 L74.0,175.0 L78.8,174.0 L83.6,173.0 L88.4,172.0 L93.2,171.0 L98.0,170.0 L102.8,169.0 L107.6,168.0 L112.4,167.0 L117.2,166.0 L122.0,165.0 L126.8,164.0 L131.6,163.0 L136.4,162.0 L141.2,161.0 L146.0,160.0 L150.8,159.0 L155.6,158.0 L160.4,157.0 L165.2,156.0 L170.0,155.0 L174.8,154.0 L179.6,153.0 L184.4,152.0 L189.2,151.0 L194.0,150.0 L198.8,149.0 L203.6,148.0 L208.4,147.0 L213.2,146.0 L218.0,145.0 L222.8,144.0 L227.6,143.0 L232.4,142.0 L237.2,141.0 L242.0,140.0 L246.8,139.0 L251.6,138.0 L256.4,137.0 L261.2,136.0 L266.0,135.0 L270.8,134.0 L275.6,133.0 L280.4,132.0 L285.2,131.0 L290.0,130.0" class="mute w2"/><path d="M50.0,30.0 L54.8,32.0 L59.6,34.0 L64.4,36.0 L69.2,38.0 L74.0,40.0 L78.8,42.0 L83.6,44.0 L88.4,46.0 L93.2,48.0 L98.0,50.0 L102.8,52.0 L107.6,54.0 L112.4,56.0 L117.2,58.0 L122.0,60.0 L126.8,62.0 L131.6,64.0 L136.4,66.0 L141.2,68.0 L146.0,70.0 L150.8,72.0 L155.6,74.0 L160.4,76.0 L165.2,78.0 L170.0,80.0 L174.8,82.0 L179.6,84.0 L184.4,86.0 L189.2,88.0 L194.0,90.0 L198.8,92.0 L203.6,94.0 L208.4,96.0 L213.2,98.0 L218.0,100.0 L222.8,102.0 L227.6,104.0 L232.4,106.0 L237.2,108.0 L242.0,110.0 L246.8,112.0 L251.6,114.0 L256.4,116.0 L261.2,118.0 L266.0,120.0 L270.8,122.0 L275.6,124.0 L280.4,126.0 L285.2,128.0 L290.0,130.0" class="ca w2"/><text x="360" y="16" class="sm">Boltzmann (softmax)</text><path d="M360,180 L600,180" class="mute"/><text x="354" y="184" text-anchor="end" class="sm">0</text><path d="M360,105.0 L600,105.0" class="mute"/><text x="354" y="109.0" text-anchor="end" class="sm">0.5</text><path d="M360,30 L600,30" class="mute"/><text x="354" y="34" text-anchor="end" class="sm">1</text><text x="360.0" y="196" text-anchor="middle" class="sm">0.05</text><text x="406.06060606060606" y="196" text-anchor="middle" class="sm">1</text><text x="454.54545454545456" y="196" text-anchor="middle" class="sm">2</text><text x="503.03030303030306" y="196" text-anchor="middle" class="sm">3</text><text x="551.5151515151515" y="196" text-anchor="middle" class="sm">4</text><text x="600.0" y="196" text-anchor="middle" class="sm">5</text><text x="480.0" y="212" text-anchor="middle" class="sm">sıcaklık T</text><path d="M360.0,180.0 L362.4,180.0 L364.8,180.0 L367.3,180.0 L369.7,179.9 L372.1,179.8 L374.5,179.5 L377.0,179.0 L379.4,178.3 L381.8,177.4 L384.2,176.4 L386.7,175.2 L389.1,174.0 L391.5,172.7 L393.9,171.3 L396.4,170.0 L398.8,168.7 L401.2,167.5 L403.6,166.2 L406.1,165.1 L408.5,163.9 L410.9,162.8 L413.3,161.8 L415.8,160.8 L418.2,159.9 L420.6,159.0 L423.0,158.1 L425.5,157.3 L427.9,156.5 L430.3,155.8 L432.7,155.1 L435.2,154.4 L437.6,153.8 L440.0,153.1 L442.4,152.6 L444.8,152.0 L447.3,151.5 L449.7,151.0 L452.1,150.5 L454.5,150.0 L457.0,149.6 L459.4,149.1 L461.8,148.7 L464.2,148.3 L466.7,148.0 L469.1,147.6 L471.5,147.2 L473.9,146.9 L476.4,146.6 L478.8,146.3 L481.2,146.0 L483.6,145.7 L486.1,145.4 L488.5,145.1 L490.9,144.9 L493.3,144.6 L495.8,144.4 L498.2,144.1 L500.6,143.9 L503.0,143.7 L505.5,143.5 L507.9,143.2 L510.3,143.0 L512.7,142.8 L515.2,142.7 L517.6,142.5 L520.0,142.3 L522.4,142.1 L524.8,141.9 L527.3,141.8 L529.7,141.6 L532.1,141.5 L534.5,141.3 L537.0,141.2 L539.4,141.0 L541.8,140.9 L544.2,140.7 L546.7,140.6 L549.1,140.5 L551.5,140.3 L553.9,140.2 L556.4,140.1 L558.8,140.0 L561.2,139.9 L563.6,139.7 L566.1,139.6 L568.5,139.5 L570.9,139.4 L573.3,139.3 L575.8,139.2 L578.2,139.1 L580.6,139.0 L583.0,138.9 L585.5,138.8 L587.9,138.7 L590.3,138.6 L592.7,138.6 L595.2,138.5 L597.6,138.4 L600.0,138.3" class="cb w2"/><path d="M360.0,180.0 L362.4,180.0 L364.8,180.0 L367.3,179.9 L369.7,179.6 L372.1,179.0 L374.5,178.0 L377.0,176.6 L379.4,174.9 L381.8,173.0 L384.2,171.0 L386.7,169.0 L389.1,167.0 L391.5,165.0 L393.9,163.2 L396.4,161.4 L398.8,159.7 L401.2,158.2 L403.6,156.7 L406.1,155.4 L408.5,154.1 L410.9,152.9 L413.3,151.9 L415.8,150.9 L418.2,149.9 L420.6,149.1 L423.0,148.3 L425.5,147.5 L427.9,146.8 L430.3,146.2 L432.7,145.6 L435.2,145.0 L437.6,144.5 L440.0,144.0 L442.4,143.5 L444.8,143.0 L447.3,142.6 L449.7,142.2 L452.1,141.9 L454.5,141.5 L457.0,141.2 L459.4,140.8 L461.8,140.5 L464.2,140.3 L466.7,140.0 L469.1,139.7 L471.5,139.5 L473.9,139.2 L476.4,139.0 L478.8,138.8 L481.2,138.6 L483.6,138.4 L486.1,138.2 L488.5,138.0 L490.9,137.9 L493.3,137.7 L495.8,137.5 L498.2,137.4 L500.6,137.2 L503.0,137.1 L505.5,136.9 L507.9,136.8 L510.3,136.7 L512.7,136.6 L515.2,136.4 L517.6,136.3 L520.0,136.2 L522.4,136.1 L524.8,136.0 L527.3,135.9 L529.7,135.8 L532.1,135.7 L534.5,135.6 L537.0,135.5 L539.4,135.4 L541.8,135.4 L544.2,135.3 L546.7,135.2 L549.1,135.1 L551.5,135.1 L553.9,135.0 L556.4,134.9 L558.8,134.8 L561.2,134.8 L563.6,134.7 L566.1,134.7 L568.5,134.6 L570.9,134.5 L573.3,134.5 L575.8,134.4 L578.2,134.4 L580.6,134.3 L583.0,134.3 L585.5,134.2 L587.9,134.1 L590.3,134.1 L592.7,134.1 L595.2,134.0 L597.6,134.0 L600.0,133.9" class="mute w2"/><path d="M360.0,30.0 L362.4,30.0 L364.8,30.0 L367.3,30.1 L369.7,30.4 L372.1,31.2 L374.5,32.5 L377.0,34.4 L379.4,36.8 L381.8,39.6 L384.2,42.6 L386.7,45.8 L389.1,49.1 L391.5,52.3 L393.9,55.5 L396.4,58.6 L398.8,61.5 L401.2,64.4 L403.6,67.0 L406.1,69.6 L408.5,72.0 L410.9,74.2 L413.3,76.3 L415.8,78.3 L418.2,80.2 L420.6,82.0 L423.0,83.6 L425.5,85.2 L427.9,86.7 L430.3,88.1 L432.7,89.4 L435.2,90.6 L437.6,91.8 L440.0,92.9 L442.4,94.0 L444.8,95.0 L447.3,95.9 L449.7,96.8 L452.1,97.7 L454.5,98.5 L457.0,99.3 L459.4,100.0 L461.8,100.7 L464.2,101.4 L466.7,102.1 L469.1,102.7 L471.5,103.3 L473.9,103.9 L476.4,104.4 L478.8,104.9 L481.2,105.4 L483.6,105.9 L486.1,106.4 L488.5,106.8 L490.9,107.3 L493.3,107.7 L495.8,108.1 L498.2,108.5 L500.6,108.9 L503.0,109.2 L505.5,109.6 L507.9,109.9 L510.3,110.3 L512.7,110.6 L515.2,110.9 L517.6,111.2 L520.0,111.5 L522.4,111.8 L524.8,112.0 L527.3,112.3 L529.7,112.6 L532.1,112.8 L534.5,113.1 L537.0,113.3 L539.4,113.5 L541.8,113.8 L544.2,114.0 L546.7,114.2 L549.1,114.4 L551.5,114.6 L553.9,114.8 L556.4,115.0 L558.8,115.2 L561.2,115.4 L563.6,115.5 L566.1,115.7 L568.5,115.9 L570.9,116.1 L573.3,116.2 L575.8,116.4 L578.2,116.5 L580.6,116.7 L583.0,116.8 L585.5,117.0 L587.9,117.1 L590.3,117.3 L592.7,117.4 L595.2,117.5 L597.6,117.7 L600.0,117.8" class="ca w2"/><text x="50" y="228" class="sm">Q = (1, 1.5, 3); mavi: en iyi eylem (3), turuncu: en kötü (1), gri: orta (1.5).</text><text x="50" y="246" class="sm">ε-açgözlü keşifte iki kötü eylemi eşit seçer (ε/3); Boltzmann değere göre ayırır: T = 1'de 0.100, 0.164, 0.736.</text></svg><figcaption>Şekil 2 — İki keşif kuralı. ε-açgözlü, rastgele adımda değere bakmaz; Boltzmann, olasılığı değerin üsteliyle orantılı verir ve T → 0'da açgözlüye, T → ∞'da düzgüne gider (hesaplandı).</figcaption></figure>

:::uyarı Kaynak düzeltmesi (lect_6): Q-öğrenmenin yakınsama koşulları
Slayt, Q-öğrenmenin en iyi Q değerlerine yakınsaması için şu koşulları sayıyor: her durum sonsuz kez ziyaret edilmeli, eylem seçimi zaman sonsuza giderken açgözlü hâle gelmeli ve öğrenme oranı Robbins–Monro koşullarını sağlamalı. İki düzeltme gerekir:

1. Ziyaret koşulu durumlar için değil, **her durum–eylem çifti** için gereklidir. Bir eylem hiç denenmezse onun $Q$'su hiç öğrenilmez.
2. Q-öğrenme politika dışıdır. $Q^*$'a yakınsaması için davranışın açgözlüye dönmesi **gerekmez**: hedef zaten $\max$'ı kullanır. Sabit $\epsilon=0.1$ ile de yakınsar (Watkins ve Dayan 1992). "Sonsuzda açgözlü" koşulu (GLIE: sonsuz keşif ile sonsuzda açgözlülük), **SARSA**'nın en iyi politikaya yakınsaması ve davranışın kendisinin en iyi hâle gelmesi için gereklidir.
:::

### Maksimizasyon yanlılığı ve çift Q-öğrenme {#cift-q}

Q-öğrenmenin hedefi $\max_{a^+}Q(s^+,a^+)$, kestirimlerin maksimumunu alır. Her kestirim yansız olsa bile maksimumları yansız değildir:
$$E\Bigl[\max_a\hat Q(a)\Bigr]\ge\max_aE\bigl[\hat Q(a)\bigr].$$
Eşitsizlik, $\max$'ın dışbükey olmasından gelir (Jensen). Gerçek değerlerin hepsi 0 ve kestirimler bağımsız $N(0,1)$ ise iki eylemde $E[\max]=1/\sqrt\pi=0.564$, on eylemde 1.54 olur (Şekil 1). Q-öğrenme bu yukarı yanlılığı her adımda hedefe katar ve yanlılık geriye doğru yayılır.

<figure class="fig wide"><svg viewBox="0 0 640 244" role="img" aria-label="Bağımsız gürültülü kestirimlerin en büyüğünün beklenen değeri, eylem sayısına göre"><path d="M60,204.0 L420,204.0" class="mute"/><text x="54" y="208.0" text-anchor="end" class="sm">0</text><path d="M60,159.0 L420,159.0" class="mute"/><text x="54" y="163.0" text-anchor="end" class="sm">0.5</text><path d="M60,114.0 L420,114.0" class="mute"/><text x="54" y="118.0" text-anchor="end" class="sm">1</text><path d="M60,69.0 L420,69.0" class="mute"/><text x="54" y="73.0" text-anchor="end" class="sm">1.5</text><path d="M60,24.0 L420,24.0" class="mute"/><text x="54" y="28.0" text-anchor="end" class="sm">2</text><text x="60.0" y="220" text-anchor="middle" class="sm">1</text><text x="135.78947368421052" y="220" text-anchor="middle" class="sm">5</text><text x="230.52631578947367" y="220" text-anchor="middle" class="sm">10</text><text x="325.2631578947368" y="220" text-anchor="middle" class="sm">15</text><text x="420.0" y="220" text-anchor="middle" class="sm">20</text><text x="240.0" y="236" text-anchor="middle" class="sm">eylem sayısı n</text><path d="M60.0,204.0 L78.9,153.2 L97.9,127.8 L116.8,111.4 L135.8,99.3 L154.7,90.0 L173.7,82.3 L192.6,75.9 L211.6,70.3 L230.5,65.5 L249.5,61.2 L268.4,57.4 L287.4,53.9 L306.3,50.7 L325.3,47.8 L344.2,45.1 L363.2,42.5 L382.1,40.2 L401.1,38.0 L420.0,35.9" class="ca w2"/><circle cx="78.94736842105263" cy="153.22293748070194" r="3.5" class="ca-f"/><text x="84.94736842105263" y="169.22293748070194" class="sm">0.564</text><circle cx="135.78947368421052" cy="99.33319737235321" r="3.5" class="ca-f"/><text x="141.78947368421052" y="115.33319737235321" class="sm">1.163</text><circle cx="230.52631578947367" cy="65.51225422483444" r="3.5" class="ca-f"/><text x="236.52631578947367" y="81.51225422483444" class="sm">1.539</text><circle cx="420.0" cy="35.92724461815109" r="3.5" class="ca-f"/><text x="426.0" y="51.92724461815109" class="sm">1.867</text><text x="60" y="14" class="sm">E[max<tspan baseline-shift="-3" font-size="75%">a</tspan> Q<tspan dx="-0.44em" dy="-0.2em">ˆ</tspan><tspan dx="0.09em" dy="0.2em">(a)]: gerçek değerlerin hepsi 0, kestirimler bağımsız N(0, 1)</tspan></text><text x="450" y="40" class="sm">Her kestirim yansız: E[Q<tspan dx="-0.44em" dy="-0.2em">ˆ</tspan><tspan dx="0.09em" dy="0.2em">(a)] = 0.</tspan></text><text x="450" y="57" class="sm">En büyüğünün beklentisi ise > 0:</text><text x="450" y="74" class="sm">  n = 2: 1/√π = 0.564</text><text x="450" y="91" class="sm">  n = 10: 1.539</text><text x="450" y="108" class="sm">  n = 20: 1.867</text><text x="450" y="125" class="sm">Q-öğrenmenin hedefi max<tspan baseline-shift="-3" font-size="75%">a</tspan> Q(s', a)</text><text x="450" y="142" class="sm">bu yanlılığı her adımda taşır.</text><text x="450" y="159" class="sm"></text><text x="450" y="176" class="sm">Çift Q: seçim bir tabloyla,</text><text x="450" y="193" class="sm">değerlendirme bağımsız ötekiyle:</text><text x="450" y="210" class="sm">E[Q<tspan dx="-0.44em" dy="-0.2em">ˆ</tspan><tspan dx="0.09em" dy="0.2em">₂(argmax Q<tspan dx="-0.44em" dy="-0.2em">ˆ</tspan><tspan dx="0.09em" dy="0.2em">₁)] = 0.</tspan></tspan></text></svg><figcaption>Şekil 1 — Maksimizasyon yanlılığı: yansız kestirimlerin maksimumu yansız değildir; yanlılık eylem sayısıyla büyür (σ√(2 ln n) mertebesinde). Değerler sayısal integralle hesaplandı.</figcaption></figure>

**Örnek** (Sutton ve Barto 6.7). A durumunda iki eylem vardır. Sağ hemen biter (ödül 0). Sol 0 ödülle B'ye götürür. B'de birçok eylem vardır; her biri biter ve $N(-0.1,1)$ ödül verir. Soldan gitmenin beklentisi −0.1'dir, yani en iyi politika hep sağdır. Q-öğrenme başlangıçta B'deki gürültülü kestirimlerin maksimumunu pozitif bulur ve A'da solu tercih eder. On eylemli B ile ilk bölümlerde solu %90'ın üstünde seçer.

**Çift Q-öğrenme** (van Hasselt 2010) iki bağımsız tablo tutar. Her adımda rastgele biri güncellenir. Eylem seçimi güncellenen tabloyla, değerlendirme öteki tabloyla yapılır:
$$Q_1(s,a)\leftarrow Q_1(s,a)+\alpha\Bigl[r+\gamma\,Q_2\bigl(s^+,\arg\max_{a^+}Q_1(s^+,a^+)\bigr)-Q_1(s,a)\Bigr].$$
$Q_2$'nin gürültüsü $Q_1$'in seçiminden bağımsızdır. Seçilen eylemin değerinin yansız bir kestirimi elde edilir. Davranış için iki tablonun toplamı kullanılır. Derin RL'de aynı fikir Double DQN olarak kullanılır (M11.6).

::lab l2

### Monte Carlo ile kontrol: GPI ve keşif başlangıçları {#sutton-mc-kontrol}

Bu paket Sutton ve Barto (2018), s.96–103 ve 110–117 içindeki kontrol ayrıntılarını tamamlar. Getiri, ilk/her ziyaret ve politika dışı oranların türetimi [M11.4](m11-4-mc-td.html#sutton-mc-temel) içindedir. Kaynak§5.2, model olmadan yalnız $V$ bilmenin açgözlü eylemi seçmeye yetmediğini vurgular: bir-adım bakış geçiş modelini ister. **Q** ise her eylemin değerini doğrudan karşılaştırır.

Belirlenimci bir politika altında birçok durum-eylem çifti hiç gözlenmez. Denenmeyen eylem iyi mi bilinemez; yalnız mevcut en iyi eylemi seçmek yetersizdir. **Keşif başlangıçları (exploring starts, ES)** her bölümün başlangıç çiftini, her $(s,a)$ için pozitif olasılıkla seçebilme varsayımıdır. Benzetimde ayarlanabilir; gerçek bir robotu her iç durumda başlatmak genellikle mümkün değildir. Her durumda bütün eylemleri pozitif olasılıkla seçen stokastik politikalar başka bir yoldur; ilgili durumlara ulaşmayı ve yeterli ziyaretleri ayrıca gerektirir.

<figure class="fig wide"><svg viewBox="0 0 640 330" role="img" aria-label="MC Q politika GPI çevrimi ve tam değerlendirme zinciri" xmlns="http://www.w3.org/2000/svg"><defs><marker id="MC-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10" fill="var(--ink)"/></marker></defs><text x="128" y="90" text-anchor="middle" style="font-size:26px;fill:var(--ink)">π</text><text x="510" y="90" text-anchor="middle" style="font-size:26px;fill:var(--ink)">Q</text><path d="M150,70 Q320,10 490,70" fill="none" stroke="var(--ink)" stroke-width="2" marker-end="url(#MC-arrow)"/><path d="M490,110 Q320,190 150,110" fill="none" stroke="var(--ink)" stroke-width="2" marker-end="url(#MC-arrow)"/><text x="320" y="22" text-anchor="middle" style="font-size:16px;fill:var(--ink)">değerlendirme: Q → qπ</text><text x="320" y="192" text-anchor="middle" style="font-size:16px;fill:var(--ink)">iyileştirme: π → greedy(Q)</text><text x="80" y="244" text-anchor="middle" style="font-size:17px;fill:var(--ink)">π₀: a0</text><path d="M132,240 L180,240" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#MC-arrow)"/><text x="160" y="224" text-anchor="middle" style="font-size:16px;fill:var(--ink)">E</text><text x="240" y="244" text-anchor="middle" style="font-size:17px;fill:var(--ink)">qπ₀=(0,1)</text><path d="M292,240 L340,240" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#MC-arrow)"/><text x="320" y="224" text-anchor="middle" style="font-size:16px;fill:var(--ink)">I</text><text x="400" y="244" text-anchor="middle" style="font-size:17px;fill:var(--ink)">π₁: a1=π*</text><path d="M452,240 L500,240" fill="none" stroke="var(--ink)" stroke-width="1.5" marker-end="url(#MC-arrow)"/><text x="480" y="224" text-anchor="middle" style="font-size:16px;fill:var(--ink)">E</text><text x="560" y="244" text-anchor="middle" style="font-size:17px;fill:var(--ink)">qπ₁=q*</text><text x="320" y="295" text-anchor="middle" style="font-size:16px;fill:var(--ink)">Tek durum, iki terminal eylem, ödüller 0 ve 1; γ=0.9</text></svg><figcaption>Şekil 3 — Kaynak s.97 içindeki Q–π GPI çevrimi ve E/I zinciri. Alttaki değerler tek-adımlı özgün MDP için tam hesaplandı; ideal zincirin bir sonlu örneği. Kaynağın numarasız iki şeması yeniden türetildi.</figcaption></figure>

Kaynak s.97 GPI şemasındaki değerlendirme oku $Q\approx q_\pi$, iyileştirme oku $\pi\approx\operatorname{greedy}(Q)$; iki süreç birbirinin hedefini değiştirir. İdeal politika yinelemesi zinciri $\pi_0\to q_{\pi_0}\to\pi_1\to q_{\pi_1}\to\cdots\to\pi_*\to q_*$ biçimindedir. Tam MC değerlendirmesi sonsuz veri ister. Şimdilik bunu ideal kabul edip ES altında $q_{\pi_k}$ tam biliniyorsa, kaynak (5.1) $\pi_{k+1}(s)\in\arg\max_aq_{\pi_k}(s,a)$ politika iyileştirmesidir. Bütün olası eylemleri düşünür, çevrenin modelini kullanmaz.

### Bölüm başına iyileştirme ve Blackjack kontrolü {#sutton-mc-es}

Sonsuz tam değerlendirme ideali gevşetilir; MC ES keşif başlangıçlarını korur: tam değer yakınsamasını beklemek yerine değişim yeterince küçük olduğunda değerlendirmeyi kesebiliriz; daha da ileri gidip **her bölümden sonra** iyileştirebiliriz. GPI değerlendirme/iyileştirme adımlarının tam bitmesini istemez. Bir yol tahmin hatasının büyüklüğüne ve olasılığına sınırlar koyup her aşamada bunları yeterince küçültecek kadar bölüm toplamaktır; yaklaşık doğruluk garantisi kurabilir ama küçük problemler dışında pahalı olabilir. Öteki yol değerlendirmeyi bitirmeden iyileştirmeye dönmektir: değer yinelemesi bir tarama, yerinde biçimi tek durum bazında bu etkileşimi kullanır. MC için doğal karşılık bölüm bazındadır. Kaynak s.99 MC ES kutusunun liste tutmayan eşdeğeri (Alıştırma 5.4):

```python
# policy keyfî, Q keyfî; N(s,a)=0.
for episode in exploring_start_episodes(policy):
    G = 0.0
    for t in reversed(range(len(episode))):
        s, a, r = episode[t]
        G = r + gamma * G
        if (s, a) in [(x, u) for x, u, _ in episode[:t]]:
            continue
        N[s, a] += 1
        Q[s, a] += (G - Q[s, a]) / N[s, a]
        policy[s] = stable_argmax(Q[s])
```

İlk ziyaret **çift** içindir, yalnız duruma bakılmaz. Bağ çözümü tutarlı seçilebilir. Kaynak bütün geçmiş getirileri, o sırada hangi politika kullanılmış olursa olsun biriktirir; politika değişince sayaçları sıfırlamaz. Sabit bir alt-optimal politikaya yerleşseydi, yeterli ziyaretle onun Q değerleri öğrenilecek ve iyileştirme yapılacaktı. Bu akıl yürütme bütün dizinin yakınsadığının kanıtı değildir.

:::uyarı Kaynak notu: yakınsama ve tarih
Kitap 2018, MC ES için optimal sabit noktaya kaçınılmaz yakınsamayı açık problem olarak sunup Tsitsiklis (2002) kısmi sonucuna atıf yapar. Burada bu, **baskının tarihsel ifadesi** olarak korunur. “Alt-optimal sabit nokta olamaz” savı, salınımın imkânsızlığını tek başına ispatlamaz; bugün için güncel literatür taraması iddia etmiyoruz.
:::

Kaynak Örnek 5.3 başlangıçta her 200 Blackjack durumu ve iki ilk eylemi eşit olasılıkla seçer. Sonraki eylemler politikayı izler; başlangıç politika 20/21 toplamında durur, Q başlangıcı 0 olabilir. Zorlanan 21 doğal olmayan karar durumudur. Mevcut x 3 bunun programını içerir; aşağıdaki deneyde 1000000 bölüm, iki as sınıfında çek/dur politikası ve $\max_aQ(s,a)$ yüzeyi gösterilir. Kaynak Şekil 5.2 içindeki optimal-değer yüzeyleri MC ES ile öğrenilen Q değerlerinden hesaplanır. Politika panelindeki 11 alt sınır etiketidir; karar verisi12–21 aralığındadır. Kaynak Şekil 5.2 bölüm bütçesi belirtmez; bir milyon bizim açık deney seçimimizdir.

::lab l 3

Tam-model optimumuyla JS politika uyumu 199/200, RMS değer farkı 0.0134602; Python aynı bütçede 199/200 ve 0.0137195 ölçüldü. Model referansında kullanılabilir-as 18 için krupiyeA karşısında çek,2…8 karşısında dur,9/10 karşısında çek çıkar. Kaynak Thorp (1966) “basic” stratejisiyle kullanılabilir-as panelinin en sol çentiği dışında eşleştiğini, bu farkın nedeninden emin olmadığını belirtir. Farkın sebebi diye doğrulanmamış bir açıklama uydurmuyoruz; bağımsız modelimiz verilen kurallardaki çentiği doğrular. Asız optimum durma eşikleri krupiyeA…10 sırasıyla 17,13,13,12,12,12,17,17,17,17; aslı eşikler 19,18,18,18,18,18,18,18,19,19.

### ES olmadan ε-yumuşak kontrol {#sutton-mc-soft}

Politika üstü öğrenmede veri üreten politika aynı zamanda iyileştirilir. **Yumuşak** politika bütün eylemlere pozitif olasılık verir. $m=|\mathcal A(s)|$ için **ε-yumuşak** politika her eylemde en az $\varepsilon/m$ verir. **ε-greedy**, bu tabanı verip kalan $1-\varepsilon$ kütleyi bir açgözlü eyleme koyar; bu sınıfta açgözlülüğe en yakın seçimdir. Belirlenimci optimum yerine keşfi sürdüren ε-yumuşak sınıfın en iyisi hedeflenir.

Kaynak s.101 sözde-kodu MC ES ile aynı ilk-ziyaret Q ortalamasını kullanır; ES başlangıcı yerine normal başlangıçtan ε-yumuşak politika izlenir. Her güncellemeden sonra $a^*=\arg\max_aQ(s,a)$ seçilir ve **bütün** eylem olasılıkları yenilenir:

$$\pi(a|s)=\begin{cases}1-\varepsilon+\varepsilon/m,&a=a^*,\\\varepsilon/m,&a\ne a^*.\end{cases}$$

Q keyfî, başlangıç politika ε-yumuşak, getiri listeleri boş veya sayaçlar 0; bölüm sonuna kadar oyna, $G\leftarrow R+\gamma G$ ile geriye git, ilk çiftleri ortala, sonra bu dağılımı ata. Bu, kaynak kutunun tüm döngüsüdür; yalnız bir eyleme olasılık yazıp diğer eski olasılıkları bırakmak toplamı 1 tutmaz. $\varepsilon$ sabitse sonuç keşif maliyetini taşır; ES olmadan keşif varsayımını daha gerçekçi hale getirmenin karşılığı budur.

:::türetim ε-yumuşak iyileştirme: kaynak (5.2)
$0<\varepsilon<1$ ve $\pi(a|s)\ge\varepsilon/m$ olsun. $w_a=(\pi(a|s)-\varepsilon/m)/(1-\varepsilon)$, negatif olmayan ve toplamı 1 olan ağırlıklardır. Yeni ε-greedy $\pi'$ için

$$q_\pi(s,\pi')=\frac\varepsilon m\sum_aq_\pi(s,a)+(1-\varepsilon)\max_aq_\pi(s,a)$$

$$\ge\frac\varepsilon m\sum_aq_\pi(s,a)+(1-\varepsilon)\sum_aw_aq_\pi(s,a)
=\sum_a\pi(a|s)q_\pi(s,a)=v_\pi(s).$$

Politika iyileştirme teoremi $v_{\pi'}\ge v_\pi$ verir. $\varepsilon=1$ için tek seçenek düzgün politikadır; $1-\varepsilon$ ile bölme yapılmaz.

Neden artık iyileşmeme ε-yumuşak optimumdur? Aynı durum/eylemli yeni çevre kurun: önerilen eylem $1-\varepsilon$ olasılıkla uygulanır, $\varepsilon$ olasılıkla düzgün rastgele başka bir seçimle değiştirilir. Yeni çevrede genel politikanın yapabilecekleri, eski çevrede ε-yumuşak politikaların yapabilecekleriyle aynıdır. Yeni Bellman optimum operatörü

$$\widetilde Tv(s)=\frac\varepsilon m\sum_a\sum_{s',r}p(s',r|s,a)[r+\gamma v(s')]
+(1-\varepsilon)\max_a\sum_{s',r}p(s',r|s,a)[r+\gamma v(s')].$$

İyileşme durduysa $v_\pi=\widetilde Tv_\pi$; bu optimum sabit nokta denklemiyle aynıdır. Sonlu $\gamma<1$ modelde tekliğinden $v_\pi=\widetilde v_*$; $\gamma=1$ için uygun sonlanma/tekillik koşulları ayrıca gerekir. İspat Q'nun **tam** $q_\pi$ olduğunu varsayar; sonlu örnek MC gürültüsünün bütün yakınsama analizini yerine koymaz.
:::

### Davranış keşfederken hedefin açgözlü olması {#sutton-mc-off}

Politika dışı kontrol hedef ile davranışı ayırır. Hedef tutarlı `argmax Q` ile belirlenimci olabilir; davranış bütün eylemleri denemeyi sürdürür. Her çift için sonsuz sayıda uygun getiri ve kapsama gerekir; davranış ε-yumuşak seçilebilir. $b$ bölümler arasında hatta bölüm içinde değişebilir; seçilen eylemin **o andaki olasılığı** saklanır. Güncelleme sırasında yeni politikadan eski davranış olasılığını yeniden hesaplamayın.

Kaynak s.111 kutusunun yeniden yazımı:

```python
# Q keyfî, C=0; pi=stable_argmax(Q).
for episode in episodes_from_soft_behavior():
    G, W = 0.0, 1.0
    for s, a, r, b_probability in reversed(episode):
        G = r + gamma * G
        C[s, a] += W
        Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
        pi[s] = stable_argmax(Q[s])
        if a != pi[s]:
            break
        W /= b_probability
```

Önce mevcut çiftin Q değeri güncellenir, sonra hedef iyileşir. Hedefe uymayan eylemde sonraki daha eski **tam getirilerin** hedef oranı 0 olacaktır, bu yüzden dış geriye döngü bırakılır. Uyan eylemde $\pi(a|s)=1$ olduğundan oran $1/b(a|s)$; payda güncellemesinde hedef olasılığının yazılmaması Alıştırma 5.11 sorusunun cevabıdır. Q zaten mevcut eylemi koşulladığı için ilk ağırlık 1.

Dezavantaj: yalnız bölümün sonundaki kesintisiz açgözlü kuyruğu kullanır. Uzun bölümde hedef dışı eylemler sıksa başlangıç durumlarına az veri kalır. Kaynak 2018 bunu araştırma sorunu olarak tartışır; TD ile önyükleme veya $\gamma<1$ iken indirim duyarlı önekler çözüm yollarıdır. Bu bir güncel literatür durumu iddiası değildir.

### Yarış pistinde iki tam görev ve ölçülen politikalar {#sutton-racetrack}

Alıştırma 5.12, Şekil 5.5 içindeki sol 17×32 ve sağ 32×30 ızgaralarda hızla dönüp bitişe varmayı ister. Sol 298 yol hücresi,6 başlangıç ve 6 bitiş; sağ 515 yol hücresi,23 başlangıç ve 9 bitiş içerir. Kaynak başlangıç çizgileri pembe/kırmızımsı, bitişler yeşildir; kendi hesaplanan çizimimizde başlangıç için turuncu, bitiş için yeşil kullanılır. İki hız bileşeni tam sayı 0…4; ikisi birlikte yalnız başlangıç çizgisinde 0 olabilir. Dokuz nominal ivme, her bileşende−1/0/+1 seçimidir. Sınır dışı son hız doğuran eylemler elenir; başka eyleme kırpılmaz.

Başlangıç çizgisinden düzgün rastgele konum ve sıfır hız seçilir. Her adım−1; hareket parçası bitişi geçince bölüm biter. Önce duvara değerse sıfır hızla rastgele başlangıca dönülür, **bölüm bitmez**. Yalnız son hücreyi kontrol etmek duvarı atlayabilir veya bitişten sonra gelen dış hücreyi yanlış çarpışma sayabilir. İki kesişimin zaman sırasını hesaplarız. Hücre köşesinde duvara temas çarpışma, bitiş yeşil hücre merkezlerinin düşey doğrusu olarak seçildi; kaynak bu sürekli geometri ayrıntısını tekilleştirmediği için seçim açıktır. Eğitimde her adım %10 olasılıkla iki ivme birden 0 olur; yörünge gösteriminde bu gürültü kapatılır, çarpışma yeniden başlangıcı korunur.

::lab l 4

Yeni x 4 aşaması bütün geometriyi, segment fiziğini, gerçek bölümleri, weighted politika dışı MC ve yörünge üretimini içerir. Varsayılan 50 bin bölümde iki pist de çalıştırıldı. Q başlangıcı−1000; başlangıç bağ kuralı düşük hızlı hücre-uzaklığı sürücüsüdür. Başlangıç çizgisinde iki son hızı 0 yapan eylemler hedefte elenir, davranışta hâlâ denenebilir. Sıfır hızdan beklemek aynı duruma−1 ile döner; diğer başlangıç hızlarında bu ek sürücü kuralıdır. Bütün yasal eylemli tam-model çözümüyle karşılaştırmada bu iki haritanın tüm optimum V değerleri değişmedi. Bu önsel tam stokastik MDP optimumundan kopyalanmış değildir. Ağırlık güncellemesi log uzayında tam $W/C$ oranını korur; önem ağırlığını kırpmaz, bütçe aşımını sahte terminale dönüştürmez.

Sol pist 50 bin bölümde 1049077 gerçek adım kullandı; bağımsız tam-model optimum başlangıç değeri−12.112482, öğrenilmiş hedef−14.458549. Kaynak “optimal politika” ister; sonlu MC çözümünü optimal diye adlandırmıyoruz. Deneyde model optimumu da seçilip aynı başlangıçlarda gösterilir, model artığı ve değer farkıyla kontrol edilir. Sağ pist 784925 gerçek adım kullandı; model optimumu−8.951270, öğrenilmiş hedef−11.447987. İki model çözümünün en büyük Bellman artığı 10⁻¹⁰ altında. Tam diziler üretim kaydındadır. Üç çizim sol/orta/sağ başlangıcı gösterir; bütün başlangıçlar ayrıca sayısal denetlendi. Gürültüsüz yörünge adım sayısı, eğitimde gürültü altında beklenen değerin negatifiyle aynı ölçü değildir.

### İndirim duyarlı kontrol uyarlaması {#sutton-mc-discount}

Alıştırma 5.14, M11.4 içindeki indirim duyarlı kestirimi kontrole taşır. Q için mevcut eylem koşullu olduğundan $\rho_{t+1:h-1}$ kullanın; boş çarpım 1. Her geriye adımda, o andaki hedef için

$$N_t=\sum_{h=t+1}^{T}c_{t,h}\rho_{t+1:h-1}\bar G_{t:h},\qquad
D_t=\sum_{h=t+1}^{T}c_{t,h}\rho_{t+1:h-1}$$

hesaplayın. $D_t>0$ ise $C(s,a)\leftarrow C(s,a)+D_t$ ve

$$Q(s,a)\leftarrow Q(s,a)+\frac{N_t-D_tQ(s,a)}{C(s,a)}$$

yapıp hedefi yeniden açgözlü seçin. İç ufuk çarpımı 0 olduğunda daha uzun önekler atlanabilir; **dış geriye döngü γ<1 için durdurulmaz**, çünkü daha kısa önekler sonraki uyumsuz eylemi içermeyebilir. γ=0 yalnız ilk ödül, γ=1 eski tam-son-ek algoritmasını verir. Yeni x 5, değişen hedefi her t için yeniden hesaplayan açık O(T²) çözüm ve bu sınırların otomatik denetimidir. Sabit hedefte O(T) yineleme M11.4 içinde türetildi. Davranış olasılığını kayıt anından alın; böyle bir cebirsel uyarlama bütün kontrol problemlerinde kendiliğinden yakınsama teoremi sağlamaz.

### MC kontrolünün tarihsel bağlantıları {#sutton-mc-kontrol-tarih}

Kitap MC ES adını 1998 ilk baskısına bağlar; MC tahminiyle politika yinelemesine dayalı kontrol arasındaki ilk açık bağlantı **olabileceğini** söyler, kesin öncelik iddiası kurmaz. Michie–Chambers (1968) çubuk dengelemede her durum/eylem için beklenen dengeleme ömrünü bölüm sürelerini ortalayarak kestirip eylem seçimini bunlarla yönetiyordu; her-ziyaret MC ES ruhuna benzer. Narendra–Wheeler (1986), ergodik sonlu Markov zincirinde aynı duruma ardışık gelişler arasındaki getiriyi öğrenme otomatının eylem olasılıklarını ayarlayan ödül olarak kullandı. Yarış pisti Barto–Bradtke–Singh (1995) ve Gardner (1973) çalışmalarından uyarlanmıştır.

Bölümün kontrol özeti: ES başlangıç çiftlerini kapsar; politika üstü yaklaşım keşfi sürdüren politikanın en iyisini arar; politika dışı yaklaşım keşif verisiyle ayrı açgözlü hedef öğrenir. Örneklerden hesaplama ve önyüklemesizlik iki ayrı tasarım tercihidir. Kaynak 2018 içindeki açık araştırma ifadeleri tarihsel olarak korunur.

### Özet {#ozet}

- Kontrol için $V$ yerine $Q(s,a)$ öğrenilir: açgözlü politika $\arg\max_aQ$ ile model olmadan çıkar. $Q^\pi$'de ilk geçiş $a$ ile koşullanır.
- $Q^*$ Bellman denkleminde maksimum beklentinin içindedir; $V^*=\max_aQ^*$, $\pi^*=\arg\max_aQ^*$.
- MC kontrol tam getirilerle $Q$'yu değerlendirir ve açgözlü iyileştirir; keşifçi başlangıçlar ya da keşif kuralı gerekir. Sayaçlar $n(s,a)$'dır.
- Q-öğrenme politika dışıdır: hedef $r+\gamma\max Q(s^+,\cdot)$. SARSA politika üstüdür: hedef $r+\gamma Q(s^+,a^+)$. Beklenen SARSA beklentiyi alır. Uçurum yürüyüşünde Q-öğrenme kenardan giden en kısa yolu, SARSA güvenli yolu öğrenir.
- Keşif: ε-açgözlü (azaltılarak), Boltzmann, iyimser başlangıç, belirsizlik bonusu.
- Q-öğrenme her $(s,a)$ sonsuz kez ziyaret edilirse ve Robbins–Monro koşullarıyla $Q^*$'a yakınsar; davranışın açgözlü olması gerekmez. GLIE SARSA için gereklidir.
- $E[\max\hat Q]\ge\max E[\hat Q]$: Q-öğrenme yukarı yanlıdır. Çift Q-öğrenme seçim ve değerlendirmeyi iki bağımsız tabloya ayırarak yanlılığı giderir.
