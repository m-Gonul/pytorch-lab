### Fonksiyonel bakış {#fonksiyonel}

`loss.backward()` bir **durum** değiştirir: grafı gezer ve parametrelerin `.grad` alanlarına yazar. Bu, standart eğitim için uygundur ama bazı soruları sormayı zorlaştırır. Her örneğin gradyanı ayrı ayrı nedir? Çıktının girdiye göre tam Jacobian'ı nedir? Hessian'ı bir vektörle nasıl çarparım? `torch.func` (eski adıyla functorch, JAX'tan esinlenir) bunlara **fonksiyon dönüşümleriyle** cevap verir. Bir fonksiyon alırlar ve yeni bir fonksiyon döndürürler.

- `grad(f)` → $\nabla f$'yi hesaplayan bir fonksiyon (f skaler döndürmeli).
- `vjp(f, x)` → $(f(x),\ \mathbf u\mapsto\mathbf u^\top\mathbf J)$; `jvp(f, (x,), (v,))` → $(f(x),\ \mathbf J\mathbf v)$.
- `jacrev(f)`, `jacfwd(f)` → tam Jacobian; `hessian(f)` = `jacfwd(jacrev(f))`.
- `vmap(f)` → $f$'yi bir batch ekseni üzerinde döngüsüz uygulayan fonksiyon.

Dönüşümler **birleştirilebilir**: `vmap(grad(f))` örnek başı gradyan, `jvp(grad(f), …)` Hessian-vektör çarpımı, `jacfwd(jacrev(f))` Hessian'dır. Bunun için $f$ **saf** olmalıdır: çıktısı yalnız girdilerine bağlı olmalı, dışarıdaki bir durumu değiştirmemelidir.

**Modeli parametrelerin fonksiyonu yapmak.** Bir `nn.Module` parametrelerini içinde tutar. `torch.func.functional_call(model, params, (x,))`, modülü verilen `params` sözlüğüyle çalıştırır. Böylece kayıp $\ell(\boldsymbol\theta,\mathbf x,y)$ biçiminde saf bir fonksiyon olur ve `grad` onun $\boldsymbol\theta$'ya göre türevini alabilir. Birden çok modeli tek seferde çalıştırmak için `stack_module_state` + `vmap` (model topluluğu) aynı fikrin uzantısıdır.

### İleri mod: dual sayılar {#dual}

M6.1'de ters modu gördük. İleri mod (forward-mode AD) türevi değerle **birlikte**, girdiden çıktıya taşır.

:::türetim Dual sayılarla JVP
$\varepsilon^2=0$ olan bir $\varepsilon$ sembolü düşünelim (dual sayılar: $a+b\varepsilon$). Çarpım $(a+b\varepsilon)(c+d\varepsilon)=ac+(ad+bc)\varepsilon$; $\varepsilon$ katsayısı tam olarak çarpım kuralıdır. Düzgün bir $g$ için Taylor açılımı birinci terimde kesilir: $g(a+b\varepsilon)=g(a)+g'(a)\,b\,\varepsilon$. Dolayısıyla her temel işlemi dual sayılar üzerinde tanımlayıp $f(\mathbf x+\mathbf v\varepsilon)$'yi hesaplarsak
$$f(\mathbf x+\mathbf v\varepsilon)=f(\mathbf x)+(\mathbf J\mathbf v)\,\varepsilon$$
elde ederiz: $\varepsilon$ kısmı JVP'dir. Hiçbir şey saklanmaz; teğet, değerle birlikte akar. Aşama b bunu saf Python'da yazıyor.
:::

El hesabı: $f(x_1,x_2)=(x_1x_2,\ x_1^2)$, $\mathbf x=(1,2)$, $\mathbf J=\begin{bmatrix}x_2&x_1\\2x_1&0\end{bmatrix}=\begin{bmatrix}2&1\\2&0\end{bmatrix}$. $\mathbf v=(1,1)$ için $\mathbf J\mathbf v=(\mathbf 3,2)$; $\mathbf u=(1,1)$ için $\mathbf u^\top\mathbf J=(\mathbf 4,1)$. İkisi de `torch.func` ile doğrulandı.

### Hangi mod? Maliyet asimetrisi {#maliyet}

<figure class="fig wide"><svg viewBox="0 0 640 238" role="img" aria-label="İleri mod Jacobian zincirini girdiden çıktıya vektörle çarpar ve bir sütun verir; ters mod çıktıdan girdiye satır vektörüyle çarpar ve bir satır verir"><defs><marker id="f1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker><marker id="f2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><text x="10" y="18" class="ta">İleri mod (JVP): J v = J₃ (J₂ (J₁ v))</text><rect x="10" y="40" width="18" height="60" rx="3" class="soft-a"/><text x="19.0" y="74.0" text-anchor="middle" class="sm">v</text><rect x="60" y="50" width="60" height="40" rx="3" class="soft-p"/><text x="90.0" y="74.0" text-anchor="middle" class="sm">J₁ (h×n)</text><rect x="140" y="30" width="80" height="80" rx="3" class="soft-p"/><text x="180.0" y="74.0" text-anchor="middle" class="sm">J₂ (h×h)</text><rect x="240" y="30" width="80" height="80" rx="3" class="soft-p"/><text x="280.0" y="74.0" text-anchor="middle" class="sm">J₃ (m×h)</text><path d="M28,70 L58,70" class="ca" marker-end="url(#f1)"/><path d="M120,70 L138,70" class="ca" marker-end="url(#f1)"/><path d="M220,70 L238,70" class="ca" marker-end="url(#f1)"/><path d="M320,70 L338,70" class="ca" marker-end="url(#f1)"/><rect x="340" y="30" width="18" height="80" rx="3" class="soft-a"/><text x="349.0" y="74.0" text-anchor="middle" class="sm">Jv</text><text x="380" y="52" class="sm">girdiden çıktıya, ileri geçişle birlikte;</text><text x="380" y="68" class="sm">her geçiş J’nin bir sütunu. Saklama yok.</text><text x="380" y="84" class="sm">Tam J: n geçiş (jacfwd).</text><text x="10" y="138" class="tb">Ters mod (VJP): uᵀJ = ((uᵀJ₃) J₂) J₁</text><rect x="10" y="160" width="80" height="18" rx="3" class="soft-b"/><text x="50.0" y="173.0" text-anchor="middle" class="sm">uᵀ</text><rect x="100" y="150" width="80" height="40" rx="3" class="soft-p"/><text x="140.0" y="174.0" text-anchor="middle" class="sm">J₃</text><rect x="190" y="150" width="80" height="40" rx="3" class="soft-p"/><text x="230.0" y="174.0" text-anchor="middle" class="sm">J₂</text><rect x="280" y="150" width="80" height="40" rx="3" class="soft-p"/><text x="320.0" y="174.0" text-anchor="middle" class="sm">J₁</text><path d="M90,169 L98,169" class="cb" marker-end="url(#f2)"/><path d="M180,169 L188,169" class="cb" marker-end="url(#f2)"/><path d="M270,169 L278,169" class="cb" marker-end="url(#f2)"/><path d="M360,169 L372,169" class="cb" marker-end="url(#f2)"/><rect x="374" y="160" width="60" height="18" rx="3" class="soft-b"/><text x="404.0" y="173.0" text-anchor="middle" class="sm">uᵀJ</text><text x="446" y="166" class="sm">çıktıdan girdiye, geri geçişle;</text><text x="446" y="182" class="sm">her geçiş bir satır. Ara değerler</text><text x="446" y="198" class="sm">saklanmalı. Tam J: m geçiş (jacrev).</text><text x="10" y="228" class="sm">El hesabı: f(x) = (x₁x₂, x₁²), x = (1, 2) → J = [[2, 1], [2, 0]]; v = (1, 1) için Jv = (3, 2); u = (1, 1) için uᵀJ = (4, 1).</text></svg><figcaption>Şekil 1 — Aynı Jacobian zinciri, iki çarpma sırası. Hangi sıranın ucuz olduğu yalnız uçlardaki boyutlara bağlıdır: Jacobian'ın sütunu (n) mu satırı (m) mı az?</figcaption></figure>

$f:\mathbb R^n\to\mathbb R^m$ ve bir ileri geçişin maliyeti $F$ olsun.
- **jacfwd**: Jacobian'ın her sütunu bir JVP'dir: $n$ JVP, toplam $\approx(1+n)F$. Ara değer saklamaz.
- **jacrev**: her satır bir VJP'dir: $m$ VJP, toplam $\approx(1+m)F$. İleri geçişin aktivasyonlarını saklar.

Kural: **girdi azsa ileri, çıktı azsa ters**. Eğitimde $m=1$ (skaler kayıp), $n$ milyonlar: ters mod tartışmasız kazanır. Ama örneğin bir robotun eklem açılarından (n = 7) uç nokta konumuna (m = 3) ya da bir ODE çözücünün az sayıda parametreye duyarlılığında ileri mod da rekabetçidir. Kare Jacobian'larda iki yol eşittir. `torch.func` her iki durumda da $n$ ya da $m$ geçişi `vmap` ile **tek bir batch'lenmiş geçişte** yapar; bu yüzden pratikteki fark işlem sayısından çok bellek ve paralellik olur.

Aşağıdaki deney küçük bir MLP'nin Jacobian'ını iki yoldan da **gerçekten** hesaplıyor. İki sonuç $10^{-16}$ hassasiyetle aynı; işlem sayısı ise kuralı izliyor (n = 8, m = 64: jacfwd 29 952, jacrev 216 320 çarpma–toplama).

::lab l1

### vmap: döngüsüz batch {#vmap}

`vmap(f, in_dims=…)` "f'yi bir örnek için yaz, batch'e kendisi uygulansın" der. Anlamı basittir: $\mathrm{vmap}(f)(\mathbf X)=\mathrm{stack}_i\,f(\mathbf X_i)$. Ama gerçekleştirimi bir Python döngüsü değildir. Her PyTorch işleminin bir **batch kuralı** vardır; vmap bu kurallarla işlemleri batch ekseni eklenmiş tek bir çağrıya çevirir (örneğin vektör–vektör iç çarpımını matris–vektör çarpımına).

- `in_dims` hangi argümanın hangi ekseninin batch olduğunu söyler (`None` = paylaşılan, ör. parametreler); `out_dims` çıktınınkini.
- **Sınırlar.** Batch'lenmiş bir değere bağlı Python kontrol akışı (`if x.sum() > 0`), `.item()`, batch'lenmemiş bir tensöre yerinde yazma ve veri bağımlı şekiller desteklenmez. Rastgelelik için `randomness="different"/"same"` seçilir.

### Örnek başı gradyanlar {#ornekbasi}

Sıradan geri geçiş batch'in **toplam** (ya da ortalama) gradyanını verir; tek tek katkılar toplanırken kaybolur. Örnek başı gradyanlar şunlar için gerekir: diferansiyel gizlilik (aşağıda), örnek önem ölçümü, gradyan çeşitliliği analizi, etki fonksiyonları.

```python
params = {k: v.detach() for k, v in model.named_parameters()}
def loss_one(p, x, y):
    return F.cross_entropy(functional_call(model, p, (x[None],)), y[None])
per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)  # {ad: (B, …)}
```

El hesabı: $\ell_i(w)=(wx_i-y_i)^2$, $w=1$, örnekler $(1,3)$ ve $(2,1)$. $g_i=2(wx_i-y_i)x_i$: $g_1=\mathbf{-4}$, $g_2=\mathbf{4}$, ortalama $\mathbf0$.

<figure class="fig wide"><svg viewBox="0 0 640 218" role="img" aria-label="Örnek başı gradyanlar eksi dört ve dört; ortalama sıfır; C = 1 ile kırpılınca eksi bir ve bir olur"><defs><marker id="p1" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ca-f"/></marker><marker id="p2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="cb-f"/></marker></defs><text x="10" y="22" class="sm">örnek başı gradyanlar</text><path d="M60,62 L580,62" class="mute"/><path d="M60,59 L60,65" class="mute"/><path d="M112,59 L112,65" class="mute"/><path d="M164,59 L164,65" class="mute"/><path d="M216,59 L216,65" class="mute"/><path d="M268,59 L268,65" class="mute"/><path d="M320,59 L320,65" class="mute"/><path d="M372,59 L372,65" class="mute"/><path d="M424,59 L424,65" class="mute"/><path d="M476,59 L476,65" class="mute"/><path d="M528,59 L528,65" class="mute"/><path d="M580,59 L580,65" class="mute"/><text x="320" y="80" text-anchor="middle" class="sm">0</text><path d="M320.0,54 L112.0,54" class="ca w2" marker-end="url(#p1)"/><text x="112" y="48" text-anchor="middle" class="ta">g₁ = −4</text><path d="M320.0,54 L528.0,54" class="ca w2" marker-end="url(#p1)"/><text x="528" y="48" text-anchor="middle" class="ta">g₂ = 4</text><text x="10" y="102" class="sm">C = 1'e kırpılmış</text><path d="M60,142 L580,142" class="mute"/><path d="M60,139 L60,145" class="mute"/><path d="M112,139 L112,145" class="mute"/><path d="M164,139 L164,145" class="mute"/><path d="M216,139 L216,145" class="mute"/><path d="M268,139 L268,145" class="mute"/><path d="M320,139 L320,145" class="mute"/><path d="M372,139 L372,145" class="mute"/><path d="M424,139 L424,145" class="mute"/><path d="M476,139 L476,145" class="mute"/><path d="M528,139 L528,145" class="mute"/><path d="M580,139 L580,145" class="mute"/><text x="320" y="160" text-anchor="middle" class="sm">0</text><path d="M320.0,134 L268.0,134" class="cb w2" marker-end="url(#p2)"/><text x="268" y="128" text-anchor="middle" class="tb">−1</text><path d="M320.0,134 L372.0,134" class="cb w2" marker-end="url(#p2)"/><text x="372" y="128" text-anchor="middle" class="tb">1</text><text x="10" y="192" class="sm">ℓ = (w x − y)², w = 1, örnekler (1, 3) ve (2, 1). Ortalama gradyan 0: batch “hiçbir şey öğrenmiyor” gibi görünür,</text><text x="10" y="208" class="sm">ama her örnek güncellemeyi 4 birim çekiyor. DP-SGD bu tek tek katkıları sınırlar; ortalama onları göstermez.</text></svg><figcaption>Şekil 2 — Örnek başı gradyan ve kırpma, modülün el hesabıyla. vmap(grad(·)) her örneğin gradyanını ayrı ayrı (B × parametre) verir; sıradan geri geçiş yalnız toplamı verir.</figcaption></figure>

:::türetim Doğrusal katmanda örnek başı gradyan bir dış çarpımdır
$\mathbf z_i=\mathbf W\mathbf x_i$ ise $\partial\ell_i/\partial\mathbf W=\bar{\mathbf z}_i\mathbf x_i^\top$ (çıktı gradyanı ile girdinin dış çarpımı); batch gradyanı $\sum_i\bar{\mathbf z}_i\mathbf x_i^\top=\bar{\mathbf Z}^\top\mathbf X$'tir. Örnek başı gradyanlar bu yüzden ileri geçişteki $\mathbf x_i$'lerden ve geri geçişteki $\bar{\mathbf z}_i$'lerden ek bir geri geçiş yapmadan kurulabilir (Goodfellow 2015'in hilesi; Opacus bunu kullanır). Bedel bellektir: $B\times$ parametre sayısı. Normları ise matrisi kurmadan hesaplanabilir: $\|\bar{\mathbf z}_i\mathbf x_i^\top\|_F=\|\bar{\mathbf z}_i\|\,\|\mathbf x_i\|$.
:::

### Diferansiyel gizlilikle SGD {#dpsgd}

Bir model eğitim verisini ezberleyebilir; bir saldırgan modelin çıktılarından belirli bir kişinin verisinin eğitimde olup olmadığını çıkarabilir (üyelik çıkarımı). **Diferansiyel gizlilik** (DP), herhangi tek bir örneği ekleyip çıkarmanın eğitimin çıktı dağılımını en fazla $e^\varepsilon$ çarpanıyla değiştirmesini ister. Abadi ve ark.'nın (2016) DP-SGD'si bunu iki adımla sağlar:
1. **Kırpma**: her örneğin gradyanı $\mathbf g_i\leftarrow\mathbf g_i\cdot\min(1,C/\|\mathbf g_i\|)$. Böylece tek bir örneğin güncellemeye katkısı (duyarlılık) en fazla $C$'dir.
2. **Gürültü**: $\tilde{\mathbf g}=\frac1B\big(\sum_i\mathbf g_i+\mathcal N(0,\sigma^2C^2\mathbf I)\big)$.

Kırpma **örnek başına** yapılmalıdır, bu yüzden örnek başı gradyan şarttır. Batch ortalamasını kırpmak tek bir örneğin katkısını sınırlamaz (el örneğinde ortalama 0'dır ama her örnek 4 çeker). Toplam gizlilik kaybı $\varepsilon$, gürültü çarpanı $\sigma$, örnekleme oranı $B/N$ ve adım sayısından bir "muhasebeci" (moments accountant, Rényi DP) ile hesaplanır. Bu modül hesabın kendisine girmiyor.

:::sezgi Gürültü neden B'ye bölünür ve büyük batch neden iyidir?
Gürültünün std'si $\sigma C$, sinyal ise $B$ kırpılmış gradyanın toplamıdır. Ortalamada gürültü payı $\sigma C/B$'dir: $B$'yi büyütmek gürültüyü sinyale göre küçültür. Ama gizlilik muhasebesinde örnekleme oranı $B/N$ arttıkça adım başına gizlilik kaybı da artar. Pratikte DP eğitimi çok büyük batch'lerle ve buna göre ayarlanmış $\sigma$ ile yapılır.
:::

Aşağıdaki deney 2-B lojistik regresyonda DP-SGD'yi sıradan SGD ile karşılaştırıyor. Bu kolay problemde gürültünün doğruluk bedeli küçük çıktı. Ama ölçüm beklenmedik bir yan etki gösterdi. Veride ters etiketli, uzaktaki aykırı örnekler varsa kırpma onların büyük gradyanlarını sınırlar. %30 aykırı örnekte SGD 0.40 doğruluğa çökerken DP-SGD 0.93'te kalıyor. Kırpma burada bir **sağlamlık** aracıdır (gradyan kırpmanın örnek başı biçimi; batch düzeyindeki kırpma M3.4, M3.7 ve M4.3’te).

::lab l2

### Hessian-vektör çarpımı {#hvp}

Hessian $\mathbf H=\nabla^2f$, $n\times n$'dir; milyon parametrede kurulamaz. Ama çoğu yöntem yalnız $\mathbf H\mathbf v$ çarpımlarına ihtiyaç duyar: Newton–CG, en büyük özdeğer için kuvvet iterasyonu (M0.2; keskinlik ve "kararlılığın kıyısı" analizleri), etki fonksiyonları.

:::türetim Hv, gradyanın yönlü türevidir
$\mathbf H\mathbf v=\frac{d}{d\epsilon}\nabla f(\mathbf x+\epsilon\mathbf v)\Big|_{\epsilon=0}$. Bu, $\nabla f$ fonksiyonunun $\mathbf v$ yönündeki JVP'sidir:
$$\mathbf H\mathbf v=\mathrm{jvp}\big(\mathrm{grad}(f),\ (\mathbf x,),\ (\mathbf v,)\big)[1].$$
**İleri-üstüne-ters** (forward-over-reverse) denir: içte bir ters geçiş (gradyan), dışta bir ileri mod türevi. Maliyeti bir gradyanın sabit katıdır (Pearlmutter 1994), bellek de öyle. Alternatif **ters-üstüne-ters**: $\nabla_{\mathbf x}\langle\nabla f(\mathbf x),\mathbf v\rangle$ (`create_graph=True` ile iki geri geçiş). Sonuç aynıdır ama ilk geri geçişin grafı da saklanır. El hesabı: $g(\mathbf x)=x_1^2x_2$, $\mathbf H=\begin{bmatrix}2x_2&2x_1\\2x_1&0\end{bmatrix}=\begin{bmatrix}4&2\\2&0\end{bmatrix}$, $\mathbf v=(1,0)$: $\mathbf H\mathbf v=(\mathbf 4,2)$.
:::

**Lojistik regresyonda HVP** (aşama c). $\mathbf H=\frac1N\mathbf X^\top\mathrm{diag}\big(p(1-p)\big)\mathbf X$. Matrisi kurmadan sağdan sola çarpılır: $\mathbf H\mathbf v=\frac1N\mathbf X^\top\big(p(1-p)\odot\mathbf X\mathbf v\big)$. Maliyet $O(Nd)$ olur, $O(Nd^2)$ değil. HVP'nin özü, "matrisi kurma, işlemlerin sırasını değiştir" fikridir.

### PyTorch'ta {#pytorch}

| amaç | kod |
|---|---|
| saf kayıp | `functional_call(model, params, (x,))` |
| gradyan | `grad(loss)(params, x, y)`; değerle: `grad_and_value` |
| örnek başı gradyan | `vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)` |
| Jacobian | `jacrev(f)(x)` (çıktı az), `jacfwd(f)(x)` (girdi az) |
| Hessian | `hessian(f)(x)` = `jacfwd(jacrev(f))(x)` |
| HVP | `jvp(grad(f), (x,), (v,))[1]` |
| JVP / VJP | `jvp(f, (x,), (v,))`, `out, fn = vjp(f, x); fn(u)` |
| model topluluğu | `params, bufs = stack_module_state(models)`; `vmap(fmodel, in_dims=(0, 0, None))` |
| DP-SGD kütüphanesi | Opacus: `PrivacyEngine().make_private(...)` |

### Özet {#ozet}

- `torch.func` dönüşümleri saf fonksiyonlar üzerinde çalışır ve birleştirilebilir; `functional_call` modeli parametrelerin fonksiyonu yapar.
- İleri mod = dual sayılar, $f(\mathbf x+\mathbf v\varepsilon)=f(\mathbf x)+\mathbf J\mathbf v\,\varepsilon$; saklama yok. El hesabı: $\mathbf J\mathbf v=(3,2)$, $\mathbf u^\top\mathbf J=(4,1)$.
- Jacobian: jacfwd $(1+n)F$, jacrev $(1+m)F$; girdi azsa ileri, çıktı azsa ters.
- vmap: batch kurallarıyla döngüsüz; veri bağımlı kontrol akışı ve `.item()` yok.
- Örnek başı gradyan `vmap(grad(·))`; doğrusal katmanda dış çarpım (el hesabı −4, 4, ortalama 0).
- DP-SGD: örnek başı kırp (C), topla, $\mathcal N(0,\sigma^2C^2)$ ekle, $B$'ye böl. Kırpma aynı zamanda aykırı örneklere karşı sağlamlık verir (deney: %30 aykırıda 0.40'a karşı 0.93).
- HVP = gradyanın JVP'si (forward-over-reverse), maliyeti gradyanın sabit katı (el hesabı (4, 2)).
