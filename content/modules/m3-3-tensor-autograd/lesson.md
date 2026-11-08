### Skalerden tensöre {#tensor}

M3.1'deki micrograd her sayıyı bir düğüm yaptı. Gerçek motorlar (PyTorch, JAX, MiniTorch) bir düğümü bir **tensör** yapar: `Y = A @ B` tek bir düğümdür, milyonlarca skaler işlem içerse bile. Her işlem türü için motorun bilmesi gereken tek şey, **tensör düzeyinde bir geri kuraldır**: çıktının gradyanı $G=\partial L/\partial Y$ verildiğinde, girdilerin gradyanları $\partial L/\partial A$, $\partial L/\partial B$ nasıl hesaplanır?

Kayıp skaler olduğu için $\partial L/\partial A$, $A$ ile aynı shape'tedir. Kuralları türetmenin güvenilir yolu, **bir elemanın** türevini zincir kuralıyla yazıp sonucu tensör işlemlerine geri toplamaktır (M0.2'de matris çarpımı için yaptığımız gibi).

### Geri kurallar = vektör–Jacobian çarpımları {#vjp}

$Y=f(A)$, $A\in\mathbb R^n$, $Y\in\mathbb R^m$ (tensörleri düzleştirerek düşün). Zincir kuralı: $\frac{\partial L}{\partial A}=J^\top G$, yani $G^\top J$ — bir **vektör–Jacobian çarpımı (VJP)**. Kilit gözlem: **Jacobian'ı hiç kurmayız**. Eleman eleman bir işlemin Jacobian'ı $n\times n$ ama köşegendir; matris çarpımının Jacobian'ı devasa ama yapılıdır. Her işlem için VJP'yi doğrudan, Jacobian'ın yapısından yararlanarak yazarız:

| ileri | geri (VJP), $G=\partial L/\partial Y$ | Jacobian'ın yapısı |
|---|---|---|
| $Y=A+B$ | $dA=\mathrm{unb}(G)$, $dB=\mathrm{unb}(G)$ | birim (+ yayın) |
| $Y=A\odot B$ | $dA=\mathrm{unb}(G\odot B)$, $dB=\mathrm{unb}(G\odot A)$ | köşegen |
| $Y=f(A)$, eleman eleman | $dA=G\odot f'(A)$ | köşegen |
| $Y=AB$ | $dA=GB^\top$, $dB=A^\top G$ | blok yapılı (M0.2 türetimi) |
| $y=\sum_{\text{eksen }k}A$ | $dA=G$, $k$ ekseninde yayılmış (`unsqueeze` + `expand`) | toplama |
| $y=\mathrm{mean}_kA$ | $dA=G/n_k$, yayılmış | |
| $Y=A^\top$, `reshape`, `permute` | $dA$ = $G$'ye ters işlem ($G^\top$, eski shape'e reshape, ters permütasyon) | permütasyon |
| $Y=A[\text{idx}]$ (gelişmiş indeks) | $dA$ = sıfırlar, `idx` konumlarına $G$'yi **ekle** (`index_add_`/`scatter_add_`) | seçici |
| $y=\mathrm{softmax}(a)$ | $da=y\odot\big(G-(G\cdot y)\mathbf 1\big)$ | $\mathrm{diag}(y)-yy^\top$ |
| $Y=\max(A,0)$ | $dA=G\odot[A>0]$ | köşegen 0/1 |

:::türetim Toplama ve indekslemenin geri kuralları
**Toplam:** $y_i=\sum_jA_{ij}$. $\frac{\partial y_i}{\partial A_{kl}}=[i=k]$, böylece $\frac{\partial L}{\partial A_{kl}}=\sum_iG_i[i=k]=G_k$: $G$, toplanan ($j$) ekseni boyunca **kopyalanır**. İleri geçişte toplama ↔ geri geçişte yayın.
**İndeksleme:** $y=A[[0,2,0]]$ ($A$'nın 0., 2., 0. elemanları). $A_0$ iki kez seçildiği için iki çıktıya katkı verir: $dA_0=G_0+G_2$, $dA_2=G_1$, $dA_1=0$. Aynı indeks birden çok kez geçebileceği için atama değil **toplama** (`scatter_add`) gerekir. Gömme katmanının (`nn.Embedding`) gradyanı bu kuraldır: aynı kelime bir batch'te iki kez geçerse satırı iki katkı alır.
**Softmax VJP:** $J=\mathrm{diag}(y)-yy^\top$ simetrik, $J^\top G=y\odot G-y(y^\top G)=y\odot(G-(G\cdot y)\mathbf1)$: $O(n^2)$'lik matris yerine $O(n)$ işlem.
:::

### Yayının geri geçişi: unbroadcast {#unbroadcast}

İleri geçişte yayın, küçük tensörü eksik ya da boyu 1 olan eksenlerde **kopyalar** (bellekte stride 0, M0.5). Geri geçişte kopyalanan her elemanın gradyanı, kopyalarının gradyanlarının **toplamıdır** (çok değişkenli zincir kuralı: bir değer birçok yerde kullanıldı).

:::tanım unbroadcast(G, shape)
1. $G$'nin girdiden **fazla olan baştaki** eksenleri üzerinden topla (shape'ten çıkar).
2. Girdide boyu **1** olan, ama $G$'de daha uzun olan her eksen üzerinden `keepdim=True` ile topla.
Sonuç girdinin shape'indedir.
:::

En yaygın örnek: $Y=XW+b$, $X$ $(B\times n)$, $b$ $(n,)$ → $b$ batch ekseninde yayılır, $db=\sum_iG_i$ (M3.2'deki $db=\mathbf1^\top dY$). Aşağıda istediğin iki shape için yayının sonucunu ve geri geçişte hangi eksenlerin toplanacağını gör.

::lab l1

:::uyarı Sessiz yayın hataları geri geçişe de taşınır
M0.2'deki `(n,)` ile `(n, 1)` karışması ileri geçişte $(n, n)$'lik yanlış bir sonuç üretir; geri geçiş de bu yanlış grafiğin gradyanını sadakatle hesaplar — hata vermeden. Autograd yalnız yazdığını türetir, niyetini değil.
:::

### Motorun mimarisi {#mimari}

MiniTorch tarzı bir motor (modülün **c** aşamasında NumPy üzerinde yazacaksın) şu parçalardan oluşur:

```python
class Tensor:
    def __init__(self, data, _prev=(), requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev, self._backward = _prev, lambda: None
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))
        def _backward():
            self.grad  += out.grad @ other.data.T      # dA = G Bᵀ
            other.grad += self.data.T @ out.grad       # dB = Aᵀ G
        out._backward = _backward
        return out
```

micrograd ile aynı iskelet (topolojik sıra, ters gezinti, `+=` birikim); farklar:

- **Gradyan bir tensördür** ve girdiyle aynı shape'te olmalıdır — yayın varsa unbroadcast şart.
- **Saklanan ara değerler** büyüktür: `ctx.save_for_backward` ile neyin saklanacağını seçmek bellek yönetiminin temelidir. Örneğin ReLU yalnız bir maske (bool) saklayabilir; sigmoid girdi yerine çıktıyı saklar ($\sigma'=y(1-y)$).
- **Yaprak (leaf) ve ara tensörler:** yalnız yaprak tensörler (`requires_grad=True` ile kullanıcının oluşturduğu parametreler) `.grad` biriktirir; ara tensörlerin gradyanı geçicidir (`retain_grad()` ile istenebilir).
- **`requires_grad` yayılımı:** bir işlemin girdilerinden biri gradyan istiyorsa çıktı da ister; hiçbiri istemiyorsa düğüm grafiğe eklenmez (`torch.no_grad()` bütün kaydı kapatır).

**PyTorch'ta özel işlem: `torch.autograd.Function`**

```python
class StableLSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):                        # x: (..., n)
        m = x.max(-1, keepdim=True).values
        y = m + (x - m).exp().sum(-1, keepdim=True).log()
        ctx.save_for_backward(x, y)
        return y.squeeze(-1)
    @staticmethod
    def backward(ctx, g):                       # g: (...,)
        x, y = ctx.saved_tensors
        return g.unsqueeze(-1) * (x - y).exp()  # ∂LSE/∂x = softmax(x)
torch.autograd.gradcheck(StableLSE.apply, (torch.randn(3, 5, dtype=torch.float64, requires_grad=True),))
```

`gradcheck` float64 girdilerle sayısal türev karşılaştırması yapar (M0.3). Özel `Function`'lar; sayısal kararlılık (yukarıdaki gibi), bellek tasarrufu (ara değeri saklamak yerine yeniden hesaplamak) ve "düz geçişli kestirici" (straight-through estimator — ileri geçişte yuvarlama, geri geçişte birim türev; nicemlemede kullanılır) için yazılır.

### Autograd'ın kuralları ve tuzakları {#tuzaklar}

| durum | ne olur | neden |
|---|---|---|
| iki kez `backward()` | `RuntimeError: Trying to backward through the graph a second time` | geri geçiş, ara değerlerin belleğini serbest bırakır; `retain_graph=True` gerekir |
| kaydedilmiş bir tensörü yerinde değiştirmek (`x.add_(1)`) | `... modified by an inplace operation` | sürüm sayacı (M0.5) ileri geçişteki değerin bozulduğunu fark eder |
| `.grad`'ı sıfırlamamak | gradyanlar adımlar arasında toplanır | `+=` birikim (bilerek kullanılırsa: gradyan biriktirme) |
| `x.detach()` | gradyan o noktada kesilir, bellek paylaşılır | hedef ağ, durdurulmuş gradyanlı kayıplar (stop-gradient) |
| `.item()`, `.numpy()` ile hesap | grafikten çıkılır | Python sayısının geçmişi yok |
| skaler olmayan çıktıda `backward()` | `grad can be implicitly created only for scalar outputs` | VJP'nin $v$'sini ver: `y.backward(v)` |
| `torch.no_grad()` içinde eğitim adımı | gradyan yok | çıkarım/değerlendirme içindir |

### İleri mod ve ters mod, ölçerek {#modlar}

M0.3'te kuramını gördük: VJP Jacobian'ın bir **satırını**, JVP bir **sütununu** verir; her biri kabaca bir ileri geçiş maliyetindedir. $f:\mathbb R^n\to\mathbb R^m$'nin tam Jacobian'ı için ters mod $m$, ileri mod $n$ geçiş ister. Aşağıda küçük bir ağın Jacobian'ını iki yoldan gerçekten hesapla, sonuçların aynı olduğunu ve maliyetin $n$ ile $m$'ye nasıl bağlı olduğunu gör.

::lab l2

PyTorch'ta: `torch.func.jacrev` (ters, `vmap`'lenmiş VJP'ler), `torch.func.jacfwd` (ileri, `vmap`'lenmiş JVP'ler), `torch.func.jvp`, `torch.func.vjp`. **Hessian** $=$ `jacfwd(jacrev(f))`: gradyan ters modla (bir skaler çıktı), onun Jacobian'ı ileri modla (girdi kadar çıktı, girdi kadar girdi — ama ileri mod bellek dostu). Hessian–vektör çarpımı ise ileri-üzerinde-ters ile tek geçişte: `jvp(grad(f), (x,), (v,))` (M0.3, Hessian kutusu).

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| düğüm | tensör; geri kural = VJP, Jacobian kurulmaz |
| temel kurallar | $+$: unb(G); $\odot$: unb(G⊙diğeri); matmul: $GB^\top$, $A^\top G$; sum ↔ yayın; indeks ↔ scatter_add |
| unbroadcast | baştaki eksenleri topla, boyu 1 olanları keepdim ile topla |
| mimari | Tensor + `_prev` + `_backward`; yaprak `.grad` biriktirir; `save_for_backward` belleği |
| tuzaklar | grafiği iki kez geri yürütmek, yerinde değişiklik, `zero_grad`, `detach`, skaler olmayan çıktı |
| modlar | tam Jacobian: ters $m$, ileri $n$ geçiş; eğitimde $m=1$ |

**Kaynaklar:** MiniTorch (Rush), Modül 1–2 (autodiff, tensörler, yayın); Karpathy, micrograd; CS231n, "Derivatives, backpropagation, and vectorization" notları (tensör türevleri, toplama ve yayın kuralları); PyTorch "Autograd mechanics" ve "Extending PyTorch" (`torch.autograd.Function`, `gradcheck`); Baydin ve ark. (2018); Bengio, Léonard, Courville (2013), düz geçişli kestirici; `torch.func` belgeleri (`jacrev`, `jacfwd`, `hessian`).
