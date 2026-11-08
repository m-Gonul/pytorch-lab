### Problem: hangi ayırıcı doğru? {#problem}

İki sınıf doğrusal olarak ayrılabiliyorsa, onları ayıran **sonsuz sayıda** doğru vardır. Lojistik regresyon (M1.4) bunlardan birini seçer, ama ayrılabilir veride ağırlıkları sonsuza gider ve hangi doğruda duracağı başlangıca ve adım sayısına bağlıdır. **Destek vektör makinesi (SVM)** net bir ilke koyar: iki sınıfa da **olabildiğince uzak** duran doğruyu seç — en geniş "güvenlik şeridini" (marj) bırakan.

Sezgi: sınırın hemen yanında duran bir örnek, biraz gürültüyle öbür tarafa geçebilir. Marj ne kadar genişse, test verisindeki küçük sapmalara karşı sınıflandırıcı o kadar dayanıklıdır. (Kuramsal karşılığı: genelleştirme sınırları $\|w\|$'ye, yani marjın tersine bağlıdır — Vapnik.)

<figure class="fig wide"><svg viewBox="0 0 640 245" role="img" aria-label="Solda veriyi ayıran birçok doğru; sağda en geniş marjlı doğru ve destek vektörleri"><g transform="translate(10,10)"><g class="ca-f"><circle cx="40" cy="40" r="5"/><circle cx="70" cy="80" r="5"/><circle cx="30" cy="120" r="5"/><circle cx="95" cy="40" r="5"/><circle cx="110" cy="110" r="5"/></g><g class="cb-f"><circle cx="200" cy="90" r="5"/><circle cx="230" cy="150" r="5"/><circle cx="260" cy="60" r="5"/><circle cx="215" cy="180" r="5"/><circle cx="280" cy="130" r="5"/><circle cx="178" cy="165" r="5"/></g><line x1="130" y1="0" x2="175" y2="200" class="mute"/><line x1="170" y1="0" x2="125" y2="200" class="mute"/><line x1="100" y1="200" x2="220" y2="0" class="mute"/><text x="150" y="228" class="sm" text-anchor="middle">hepsi eğitim verisini ayırır</text></g><g transform="translate(330,10)"><g class="ca-f"><circle cx="40" cy="40" r="5"/><circle cx="70" cy="80" r="5"/><circle cx="30" cy="120" r="5"/><circle cx="95" cy="40" r="5"/><circle cx="110" cy="110" r="5"/></g><g class="cb-f"><circle cx="200" cy="90" r="5"/><circle cx="230" cy="150" r="5"/><circle cx="260" cy="60" r="5"/><circle cx="215" cy="180" r="5"/><circle cx="280" cy="130" r="5"/><circle cx="178" cy="165" r="5"/></g><polygon points="95,0 135,0 175,200 135,200" class="soft-p"/><line x1="115" y1="0" x2="155" y2="200" class="cp w2"/><line x1="95" y1="0" x2="135" y2="200" class="cp dash"/><line x1="135" y1="0" x2="175" y2="200" class="cp dash"/><circle cx="110" cy="110" r="9" class="cq w2" fill="none"/><circle cx="95" cy="40" r="9" class="cq w2" fill="none"/><circle cx="178" cy="165" r="9" class="cq w2" fill="none"/><text x="230" y="30" class="sm tq">destek vektörleri</text><text x="150" y="228" class="sm tp" text-anchor="middle">marj = 2/‖w‖</text></g></svg><figcaption>Şekil 1 — Sol: eğitim verisini ayıran birçok doğru vardır. Sağ: SVM, iki sınıfa en uzak olanı seçer. Şeridin kenarına değen noktalar (mor halkalı) destek vektörleridir; çözüm yalnız onlara bağlıdır.</figcaption></figure>

### Sert marjlı SVM {#sert}

Etiketler $y_i\in\{-1,+1\}$ (lojistik regresyondaki $\{0,1\}$ değil; formüller böyle sadeleşir), karar fonksiyonu $f(x)=w^\top x+b$, tahmin $\mathrm{sign}(f(x))$. Bir noktanın sınıra **işaretli** uzaklığı $\frac{y_i(w^\top x_i+b)}{\|w\|}$ (pozitif = doğru taraf; türetim aşağıdaki kutuda).

$(w,b)$'yi bir $c>0$ ile çarpmak sınırı değiştirmez. Bu ölçek serbestliğini, en yakın noktalarda $y_if(x_i)=1$ olacak biçimde sabitleriz (**kanonik biçim**). O zaman en yakın noktanın uzaklığı $1/\|w\|$, marj $2/\|w\|$ ve problem:
$$\min_{w,b}\ \tfrac12\|w\|^2\quad\text{öyle ki}\quad y_i(w^\top x_i+b)\ge1,\ \ i=1..n.$$
Dışbükey bir kuadratik program: tek bir global çözüm.

Uzaklık formülünün ve $2/\|w\|$'nin adım adım türetimi yukarıdaki giriş bölümünün kutusundadır.

### Yumuşak marj ve hinge kaybı {#yumusak}

Gerçek veri nadiren ayrılabilir ve tek bir aykırı nokta sert marjı yok edebilir. **Gevşek değişkenler** $\xi_i\ge0$ her kısıtın ne kadar ihlal edildiğini ölçer:
$$\min_{w,b,\xi}\ \tfrac12\|w\|^2+C\sum_i\xi_i\quad\text{öyle ki}\quad y_if(x_i)\ge1-\xi_i,\ \ \xi_i\ge0.$$
Optimumda her $\xi_i$ olabildiğince küçük seçilir: $\xi_i=\max(0,1-y_if(x_i))$. Yerine koyunca **kısıtsız** bir problem:
$$\min_{w,b}\ \underbrace{\frac1n\sum_i\max\big(0,1-y_if(x_i)\big)}_{\text{hinge kaybı}}+\lambda\|w\|^2,\qquad \lambda=\frac1{2nC}.$$
Yani yumuşak marjlı SVM = **L2 düzenlileştirilmiş hinge kaybı** — M1'deki "kayıp + ceza" kalıbının bir örneği. $C$ büyük: ihlaller pahalı, dar marj (sert marja yaklaşır, aşırı uyum riski); $C$ küçük: geniş marj, birçok ihlale izin (yanlılık).

**Hinge kaybının yorumu** (M1.4 kayıp karşılaştırma deneyini hatırla): marj $m=yf(x)\ge1$ olan örnekler **sıfır** kayıp ve sıfır gradyan verir — doğru tarafta ve yeterince uzaktalar, modeli hiç etkilemezler. Lojistik kayıp ise hiçbir zaman tam sıfır olmaz; her örnek az da olsa çeker. Bu fark SVM'in **seyrekliğinin** kaynağıdır: çözüm yalnız $m\le1$ olan örneklere bağlıdır.

**Alt-gradyanla eğitim.** Hinge $m=1$'de kırılır; orada türev yok, alt-gradyan var (M0.3): aktif örnekler ($y_if(x_i)<1$) için $\partial\ell_i/\partial w=-y_ix_i$, diğerleri için 0. Stokastik alt-gradyan inişi (Pegasos, Shalev-Shwartz ve ark. 2007) büyük doğrusal SVM'leri bu yolla eğitir; öğrenme oranı $\eta_t=1/(\lambda t)$ ile yakınsama kanıtlıdır.

### Dual problem ve destek vektörleri {#dual}

:::türetim Lagrange dualı
Sert marj problemi için Lagrange fonksiyonu ($\alpha_i\ge0$ çarpanlar):
$$\mathcal L(w,b,\alpha)=\tfrac12\|w\|^2-\sum_i\alpha_i\big[y_i(w^\top x_i+b)-1\big].$$
$w$ ve $b$'ye göre minimize et: $\nabla_w\mathcal L=w-\sum_i\alpha_iy_ix_i=0\Rightarrow\boxed{w=\sum_i\alpha_iy_ix_i}$; $\partial\mathcal L/\partial b=-\sum_i\alpha_iy_i=0$. Yerine koyunca **dual problem**:
$$\max_{\alpha\ge0}\ \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,x_i^\top x_j\quad\text{öyle ki}\quad\sum_i\alpha_iy_i=0.$$
Yumuşak marjda tek değişiklik: $0\le\alpha_i\le C$. **KKT tamamlayıcılık koşulu** $\alpha_i[y_if(x_i)-1+\xi_i]=0$ şunu söyler: marjın dışında kalan ($y_if(x_i)>1$) her örnek için $\alpha_i=0$. Sıfırdan farklı $\alpha$'lı örnekler **destek vektörleridir**: $0<\alpha_i<C$ olanlar tam marj üzerinde ($y_if=1$), $\alpha_i=C$ olanlar marjın içinde ya da yanlış tarafta.
:::

Dual iki şeyi açığa çıkarır: (1) çözüm **seyrek** bir örnek kombinasyonudur, $w=\sum_{i\in SV}\alpha_iy_ix_i$; (2) veri problemde **yalnız iç çarpımlar** $x_i^\top x_j$ olarak görünür. İkincisi çekirdek hilesinin kapısıdır.

### Çekirdek hilesi {#cekirdek}

Doğrusal olmayan bir sınır için öznitelikleri dönüştür: $x\mapsto\phi(x)$ ve dönüştürülmüş uzayda doğrusal SVM eğit. Dual'de $x_i^\top x_j$ yerine $\phi(x_i)^\top\phi(x_j)$ gelir. Eğer bu iç çarpımı $\phi$'yi hiç hesaplamadan doğrudan veren bir fonksiyon varsa — bir **çekirdek** $k(x,x')=\phi(x)^\top\phi(x')$ — dönüştürülmüş uzay çok büyük, hatta sonsuz boyutlu olabilir.

:::türetim Polinom çekirdeğinin öznitelik haritası
$x\in\mathbb R^2$, $k(x,z)=(x^\top z)^2=(x_1z_1+x_2z_2)^2=x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2=\phi(x)^\top\phi(z)$, $\phi(x)=(x_1^2,\sqrt2x_1x_2,x_2^2)$. $d$ boyut ve derece $p$ için $(1+x^\top z)^p$ tüm $\le p$ dereceli tek terimlileri içerir: $\binom{d+p}p$ boyut — $d=100$, $p=3$'te 176 851. Çekirdek bunu $O(d)$'de hesaplar.
:::

| çekirdek | $k(x,x')$ | öznitelik uzayı | not |
|---|---|---|---|
| doğrusal | $x^\top x'$ | $\mathbb R^d$ | büyük, seyrek veri (metin) |
| polinom | $(1+x^\top x')^p$ | $\le p$ dereceli tek terimliler | |
| RBF (Gauss) | $\exp(-\gamma\|x-x'\|^2)$ | sonsuz boyutlu | en yaygın; $\gamma$ yerellik |
| sigmoid | $\tanh(a\,x^\top x'+c)$ | — | her parametrede geçerli çekirdek değil |

**Hangi fonksiyonlar çekirdektir? (Mercer)** Her sonlu nokta kümesi için $K_{ij}=k(x_i,x_j)$ matrisi simetrik ve pozitif yarı-tanımlıysa. Çekirdekler toplanabilir, pozitif sayıyla çarpılabilir, çarpılabilir: yeni çekirdekler kurmanın kuralları.

**Temsilci teoremi (representer theorem):** "kayıp + $\|w\|^2$" biçimindeki her problemin çözümü $w=\sum_i\alpha_i\phi(x_i)$ biçimindedir; dolayısıyla $f(x)=\sum_i\alpha_ik(x_i,x)+b$. RBF ile bu, her destek vektörüne bir Gauss "tümseği" koymaktır; $\gamma$ tümseğin darlığı.

:::uyarı Çekirdek yöntemlerinin ölçek sorunu
$K$ matrisi $n\times n$'dir: $n=10^5$ örnekte 40 GB (float32), çözücüler $O(n^2)$–$O(n^3)$. Büyük veride ya doğrusal SVM, ya çekirdeği yaklaştıran **rastgele öznitelikler** (Rahimi ve Recht 2007: $\phi(x)=\sqrt{2/D}\cos(\Omega x+b)$, $\Omega$ Gauss — RBF'nin yansız yaklaşımı) ya da sinir ağları kullanılır. Derin öğrenmenin yükselişinde bu ölçeklenebilirlik farkı belirleyiciydi.
:::

Aşağıda dual problemi çözen gerçek bir çekirdek SVM: veri kümesini, çekirdeği, $C$'yi ve $\gamma$'yı değiştir; destek vektörlerinin hangi noktalar olduğunu ve sınırın nasıl değiştiğini izle.

::lab l1

### Çok sınıflı SVM {#cok-sinif}

İki yaygın yol: **bire karşı hepsi** ($K$ ikili SVM) ve **tek bir çok sınıflı kayıp**. CS231n'in kullandığı Weston–Watkins kaybı, skorlar $s=Wx$ için:
$$\ell=\sum_{j\ne y}\max(0,\ s_j-s_y+\Delta).$$
Doğru sınıfın skoru, diğer her sınıfınkinden en az $\Delta$ kadar büyük olmalı; olmayan her sınıf için aradaki açık kadar ceza. $\Delta=1$ yaygındır ($W$'nin ölçeği L2 cezasıyla birlikte $\Delta$'nın önemini ortadan kaldırır).

:::türetim Gradyan
Aktif sınıflar ($s_j-s_y+\Delta>0$) için $\partial\ell/\partial s_j=1$, doğru sınıf için $\partial\ell/\partial s_y=-(\text{aktif sınıf sayısı})$. Batch: $M=\max(0,S-s_y\mathbf1^\top+\Delta)$ ($y$ sütunları 0), $B=\mathbb1[M>0]$, $B_{i,y_i}=-\sum_jB_{ij}$, $\nabla_W=X^\top B/n$ (modülün **c** aşaması, döngüsüz).
:::

**SVM kaybı mı softmax mı?** Softmax (M1.5) her örneğin olasılığını sürekli iyileştirmeye çalışır; SVM kaybı, marj sağlandığı anda o örnekle ilgilenmeyi bırakır. Doğrulukta genelde benzerdirler; softmax kalibre edilmiş olasılık verir, SVM vermez (olasılık için Platt ölçeklemesi gerekir).

### SVM bugün {#bugun}

SVM'ler 1995–2012 arasında birçok alanda en iyi yöntemdi (el yazısı rakam, metin sınıflandırma, biyoinformatik). Bugün:

- Küçük ve orta boyutlu tablo verisinde, iyi ölçeklenmiş özniteliklerle hâlâ güçlü bir taban çizgisidir (RBF + $C,\gamma$ ızgara araması).
- Derin ağlarda **hinge benzeri** kayıplar yaşar: metrik öğrenmede marjlı kayıplar (triplet, contrastive), GAN'larda hinge kaybı, yüz tanımada açısal marjlı softmax (ArcFace).
- Çekirdek bakışı derin öğrenme kuramında geri döndü: çok geniş ağların eğitimi bir çekirdek yöntemine indirgenebilir (Neural Tangent Kernel, Jacot ve ark. 2018).

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| marj | $2/\|w\|$; en geniş marjı arayan ayırıcı |
| yumuşak marj | $\tfrac12\|w\|^2+C\sum\max(0,1-y_if(x_i))$ = L2 + hinge |
| $C$ | büyük: dar marj, az ihlal; küçük: geniş marj, çok ihlal |
| dual | $w=\sum\alpha_iy_ix_i$; $\alpha_i>0$ yalnız destek vektörlerinde |
| çekirdek | iç çarpımı $k(x,x')$ ile değiştir; RBF'de $\gamma$ yerellik |
| ölçek | $K$ $n\times n$; büyük veride doğrusal SVM ya da rastgele öznitelikler |

**Kaynaklar:** Stanford CS229 ders notları, "Support vector machines" (marj, Lagrange dualı, KKT, çekirdekler, SMO); CS231n, "Multiclass SVM loss" ve Assignment 1; Cortes ve Vapnik (1995), "Support-vector networks"; Schölkopf ve Smola, *Learning with Kernels*; Shalev-Shwartz ve ark. (2007), Pegasos; Hsieh ve ark. (2008), "A dual coordinate descent method for large-scale linear SVM"; Rahimi ve Recht (2007), "Random features for large-scale kernel machines"; Jacot ve ark. (2018), Neural Tangent Kernel.
