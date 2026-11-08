### Problem: çok boyut, az bilgi {#problem}

Bir yüz görüntüsü 64 × 64 = 4096 sayıdır, ama tüm olası 4096 boyutlu vektörlerin içinde "yüz gibi görünenler" küçük, yapılı bir bölge oluşturur: komşu pikseller birbirine çok benzer, yüzler ortak bir yapıyı paylaşır. Veri **ilişkili**dir ve gerçek serbestlik derecesi boyut sayısından çok azdır. **Temel bileşen analizi (PCA)** bu fazlalığı doğrusal olarak ortadan kaldırır: veriyi, en çok değiştiği birkaç yönde ifade eder.

Kullanım alanları: **sıkıştırma** (4096 sayı yerine 50 katsayı), **görselleştirme** (2–3 boyuta indir), **gürültü giderme** (küçük varyanslı yönler çoğunlukla gürültüdür), **ön işleme** (ilişkisiz, ölçekli öznitelikler — beyazlatma), **analiz** (bir sinir ağının ara temsillerinde kaç yön gerçekten kullanılıyor?).

### Kovaryans ve geometri {#kovaryans}

Ortalanmış veri $X_c=X-\mathbf1\bar x^\top\in\mathbb R^{n\times d}$ (satır = örnek). Kovaryans matrisi (M0.4):
$$C=\frac1{n-1}X_c^\top X_c\in\mathbb R^{d\times d},\qquad C_{jk}=\mathrm{Cov}(x_j,x_k).$$
$C$ simetrik ve pozitif yarı-tanımlıdır. Bir **birim** yön $u$ boyunca izdüşümler $z_i=u^\top(x_i-\bar x)$'in varyansı:
$$\mathrm{Var}(z)=\frac1{n-1}\|X_cu\|^2=u^\top Cu.$$
Gauss benzeri veride eş-yoğunluk eğrileri elipslerdir; elipsin eksenleri $C$'nin özvektörleri, yarı-eksen uzunlukları özdeğerlerin karekökleridir.

:::tanım Spektral ayrışım (simetrik matrisler)
Her gerçel simetrik $C$ için $C=V\Lambda V^\top$: $V$ ortonormal (sütunları özvektörler $v_1,\dots,v_d$, birbirine dik), $\Lambda=\mathrm{diag}(\lambda_1\ge\dots\ge\lambda_d)$ gerçel özdeğerler. $C$ pozitif yarı-tanımlıysa $\lambda_i\ge0$. Yani **her** kovaryans matrisi, uygun bir döndürmeden sonra köşegendir: o eksenlerde öznitelikler ilişkisizdir.
:::

### PCA'nın iki eşdeğer tanımı {#tanim}

**1. En büyük varyans:** veriyi $k$ boyuta izdüşürürken korunan varyansı en büyükle. Tek bir yön için çözüm $C$'nin en büyük özdeğerine ait özvektördür ve korunan varyans o özdeğerdir (türetimi, Lagrange çarpanıyla, giriş bölümündeki kutuda). $k$ yön için: ilk $k$ özvektör, korunan varyans $\lambda_1+\dots+\lambda_k$.

**2. En küçük yeniden kurma hatası:** $k$ boyutlu bir alt uzaya izdüşürüp geri döndürünce kare hatayı en küçükle.

:::türetim İki tanım neden aynı?
$u$ birim vektör, her nokta için $x=(u^\top x)u+r$, artık $r\perp u$. Pisagor: $\|x\|^2=(u^\top x)^2+\|r\|^2$. Tüm (ortalanmış) noktalar üzerinden toplayınca
$$\underbrace{\sum_i\|x_i\|^2}_{\text{sabit: toplam varyans}\times(n-1)}=\underbrace{\sum_i(u^\top x_i)^2}_{\text{korunan}}+\underbrace{\sum_i\|r_i\|^2}_{\text{yeniden kurma hatası}}.$$
Sol taraf $u$'ya bağlı değil: korunanı en büyüklemek, hatayı en küçüklemektir. $k$ boyutlu alt uzaylar için de aynı (alt uzaya dik izdüşüm, M1.1).
:::

Aşağıda kendin dene: bir yön seç, izdüşümleri ve artıkları gör; korunan varyans ile hata eğrilerinin birbirinin aynası olduğunu ve tepe noktasının PC1'de olduğunu izle.

::lab l1

**Kaç bileşen?** **Açıklanan varyans oranı** $\frac{\lambda_1+\dots+\lambda_k}{\sum_j\lambda_j}$. Yaygın kurallar: oran %90–95'i geçene kadar; ya da özdeğerlerin grafiğinde (scree plot) "dirsek" noktası; ya da sonraki görevdeki (sınıflandırma, yeniden kurma) doğrulama performansı. Özdeğerlerin toplamı **izdir**: $\sum_j\lambda_j=\mathrm{tr}(C)=\sum_j\mathrm{Var}(x_j)$ — toplam varyans döndürmeyle değişmez.

### SVD ile PCA {#svd}

:::tanım Tekil değer ayrışımı (SVD)
Her $M\in\mathbb R^{n\times d}$ için $M=USV^\top$: $U\in\mathbb R^{n\times r}$ ve $V\in\mathbb R^{d\times r}$ ortonormal sütunlu, $S=\mathrm{diag}(s_1\ge\dots\ge s_r>0)$, $r=\mathrm{rank}(M)$. Geometrisi (M0.2): her doğrusal dönüşüm = döndürme ($V^\top$) → eksenler boyunca ölçekleme ($S$) → döndürme ($U$).
:::

$X_c=USV^\top$ ise $X_c^\top X_c=VS^2V^\top$: **$V$'nin sütunları $C$'nin özvektörleridir** ve $\lambda_i=s_i^2/(n-1)$. Bileşen skorları $Z=X_cV_k=U_kS_k$. PCA'yı SVD ile hesaplamak, kovaryansı açıkça kurup özdeğer ayrıştırmaktan **sayısal olarak daha iyidir**: $X_c^\top X_c$'nin koşul sayısı $X_c$'ninkinin karesidir (M1.1 uyarısı). $d$ büyük, $n$ küçükse (4096 piksel, 100 görüntü) $d\times d$ kovaryansı hiç kurmamak büyük bellek kazancıdır.

:::kanıt Eckart–Young: kesik SVD en iyi rank-k yaklaşımdır (ifade)
$M_k=\sum_{i\le k}s_iu_iv_i^\top$ (M0.2'nin dış çarpım görünümü: $k$ tane rank-1 matris). Rank'ı en fazla $k$ olan her $B$ için $\|M-B\|_F\ge\|M-M_k\|_F=\sqrt{\sum_{i>k}s_i^2}$ (ve spektral normda $\ge s_{k+1}$). Kanıt fikri: $B$'nin çekirdeği en az $d-k$ boyutludur ve ilk $k+1$ sağ tekil vektörün gerdiği alt uzayla kesişir; o kesişimdeki birim vektör $x$ için $\|(M-B)x\|=\|Mx\|\ge s_{k+1}$. Frobenius sürümü benzer bir argümanla tüm atılan tekil değerlere genişler. PCA'nın "en küçük yeniden kurma hatası" tanımı, bu teoremin ortalanmış veriye uygulanmasıdır.
:::

Aşağıda bir görüntüyü matris olarak gör: rank-$k$ yaklaşımları, tekil değer spektrumunu ve gürültü giderme etkisini incele.

::lab l3

### Hesaplama: kuvvet yinelemesi {#kuvvet}

Yalnız ilk birkaç bileşen gerekiyorsa tam SVD gereksizdir. **Kuvvet yinelemesi:** rastgele bir $v_0$ ile başla, $v\leftarrow Cv/\|Cv\|$ tekrarla.

:::türetim Neden ve ne hızla yakınsar?
$v_0=\sum_ia_iv_i$ (özvektör tabanında). $C^tv_0=\sum_ia_i\lambda_i^tv_i=\lambda_1^t\big(a_1v_1+\sum_{i\ge2}a_i(\lambda_i/\lambda_1)^tv_i\big)$. $|\lambda_2|<\lambda_1$ ve $a_1\ne0$ ise parantezin içi $a_1v_1$'e, normalleştirilmiş vektör $\pm v_1$'e yakınsar; hata $|\lambda_2/\lambda_1|^t$ ile küçülür. **Koşullar:** baskın bir özdeğer olmalı ($\lambda_1>|\lambda_2|$); $\lambda_1\approx\lambda_2$ ise yakınsama çok yavaş (ve özvektör zaten kararsız: veriyi biraz değiştirmek yönü döndürür). $v_0\perp v_1$ ise ($a_1=0$) kuramsal olarak $v_1$ hiç bulunmaz.
:::

**Sonraki bileşenler — deflasyon:** $C\leftarrow C-\lambda_1v_1v_1^\top$ ile bulunan bileşeni çıkar ve tekrarla. Pratikte birden çok vektörü aynı anda yineleyip ortonormalleştirmek (alt uzay yinelemesi, Lanczos, rastgele SVD — Halko ve ark. 2011) daha hızlı ve kararlıdır. Modülün **d** aşamasında aynı problemi Rayleigh bölümü $\frac{v^\top Cv}{v^\top v}$'yu autograd ile en büyükleyerek çözeceksin.

::lab l2

### Pratik ayrıntılar {#pratik}

- **Ortalama çıkar:** PCA ortalanmış veride tanımlıdır; ortalamayı çıkarmazsan ilk bileşen genelde yalnız ortalamaya işaret eder.
- **Ölçekle mi?** Öznitelikler farklı birimdeyse (metre ve gram) büyük birimli olan varyansa hükmeder. Standartlaştırmak = **korelasyon** matrisinin PCA'sı. Aynı birimdeki özniteliklerde (pikseller) genelde ölçeklenmez.
- **İşaret belirsizliği:** $v$ bir özvektörse $-v$ de öyledir; farklı kütüphaneler farklı işaret döndürebilir. Karşılaştırma için bir kural koy (modülde: en büyük mutlak elemanı pozitif yap).
- **Aykırı değerler:** kare uzaklıklar yüzünden tek bir uzak nokta bileşenleri kendine çekebilir (birinci deneyde dene).
- **Beyazlatma (whitening):** $Z\Lambda_k^{-1/2}$ — bileşen skorlarını birim varyansa ölçekle: kovaryansı $I$ olan veri. Bazı yöntemlerin (ICA, eski öz-denetimli öğrenme) ön koşulu; küçük özdeğerli (gürültülü) yönleri büyüttüğü için dikkatli kullanılmalı.

### Sınırlar ve ötesi {#sinirlar}

- **Yalnız doğrusal yapı:** bir sarmal ya da kıvrık bir yüzey üzerindeki veriyi PCA düzleştiremez. **Çekirdek PCA** (M2.3'teki çekirdek hilesi), **otokodlayıcılar** (M5.6: doğrusal otokodlayıcının en iyi çözümü PCA alt uzayıdır), t-SNE ve UMAP gibi doğrusal olmayan yöntemler bunun için var (son ikisi yalnız görselleştirme içindir; uzaklıkları korumazlar).
- **Varyans ≠ bilgi:** sınıfları ayıran yön küçük varyanslı olabilir; PCA onu atar. Etiket varsa **LDA** (sınıflar arası / sınıf içi varyans oranını en büyükleyen yön) daha uygun olabilir.
- **Derin öğrenmede:** ağırlık matrislerinin ve gradyanların düşük **etkili rank**'ı (tekil değerleri hızla düşer) LoRA gibi düşük ranklı ince ayar yöntemlerinin (M0.2) ve sıkıştırmanın dayanağıdır; ara temsillerin PCA'sı, bir katmanın kaç boyutu gerçekten kullandığını gösterir.

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| kovaryans | $C=\frac1{n-1}X_c^\top X_c$; yön varyansı $u^\top Cu$ |
| PCA | $C$'nin özvektörleri; varyansı en büyükleme = yeniden kurma hatasını en küçükleme |
| SVD | $X_c=USV^\top$: $V$ bileşenler, $\lambda=s^2/(n-1)$, skorlar $US$ |
| Eckart–Young | kesik SVD en iyi rank-k yaklaşım; hata $\sqrt{\sum_{i>k}s_i^2}$ |
| kuvvet yinelemesi | hata $\sim|\lambda_2/\lambda_1|^t$; deflasyon ile sonrakiler |
| pratik | ortala, gerekirse standartlaştır, işareti sabitle, aykırılara dikkat |

**Kaynaklar:** Deisenroth, Faisal, Ong, *Mathematics for Machine Learning*, Bölüm 10 (PCA'nın iki türetimi); Strang, *Introduction to Linear Algebra* Bölüm 7 ve MIT 18.06 SVD dersi; James ve ark., *ISLP* §12.2; Shlens (2014), "A tutorial on principal component analysis"; Eckart ve Young (1936); Halko, Martinsson, Tropp (2011), "Finding structure with randomness" (rastgele SVD); Golub ve Van Loan, *Matrix Computations* (kuvvet yinelemesi, Jacobi yöntemi).
