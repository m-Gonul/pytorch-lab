### Neden kendi autograd'ımızı yazıyoruz? {#neden}

M0.3'te geri yayılımı kâğıt üzerinde yaptık; M1–M2'de gradyanları her model için elle türettik ($\frac2nX^\top e$, $p-y$, …). Sinir ağlarında bu sürdürülemez: yüzlerce katman, onlarca işlem türü, her değişiklikte yeni türetim. **Otomatik türev** (autograd) motoru, her işlemin yalnız **kendi yerel türevini** bilmesini ister; gerisini — zincir kuralını grafik boyunca uygulamayı — kendisi yapar.

`loss.backward()` bir kara kutu kaldığı sürece birçok şey gizemli görünür: gradyanlar neden birikir (`zero_grad`), neden bir tensörü yerinde değiştirince hata alırız, `detach` ne yapar, bellek neden ileri geçişte dolar. Bu modülde Karpathy'nin **micrograd**'ını (yaklaşık 100 satır) yeniden kuracağız; bu soruların hepsinin cevabı o 100 satırdadır.

### Value: değer + nereden geldiği + nasıl geri gidileceği {#value}

Her skaler bir `Value` nesnesine sarılır ve üç şey taşır:

```python
class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = data          # ileri geçişteki değer
        self.grad = 0.0           # ∂(çıktı)/∂(bu değer), geri geçişte dolar
        self._prev = set(_prev)   # bu değeri üreten girdiler (grafiğin kenarları)
        self._op = _op            # hata ayıklama için
        self._backward = lambda: None   # yerel türevi girdilere dağıtan kapanış (closure)
```

Her işlem yeni bir `Value` üretir ve onun `_backward`'ına "çıktının gradyanını girdilerime nasıl dağıtırım"ı yazar:

```python
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += other.data * out.grad    # ∂(a·b)/∂a = b
            other.grad += self.data  * out.grad    # ∂(a·b)/∂b = a
        out._backward = _backward
        return out
```

İki ayrıntı her şeydir:

1. **`+=`, `=` değil.** Bir değer birden çok yerde kullanılırsa (ör. `x*x + x`), her kullanım kendi katkısını ekler — çok değişkenli zincir kuralının toplamı (M0.3). `=` yazmak en yaygın micrograd hatasıdır: son yol öncekileri siler. PyTorch'un `.grad`'ı aynı nedenle **birikir** ve her eğitim adımında `zero_grad()` gerekir.
2. **Kapanış, ileri geçişin değerlerini yakalar** (`self.data`, `other.data`). Geri geçiş bu değerlere ihtiyaç duyar; bu yüzden ileri geçişteki ara değerler bellekte tutulur. Büyük ağlarda eğitim belleğinin çoğu bunlardır (M0.3, ters modun bedeli).

| işlem | ileri | `_backward` (gelen gradyan $g$ = `out.grad`) |
|---|---|---|
| $o=a+b$ | $a+b$ | `a.grad += g`, `b.grad += g` |
| $o=a\cdot b$ | $ab$ | `a.grad += b·g`, `b.grad += a·g` |
| $o=a^k$ ($k$ sabit) | $a^k$ | `a.grad += k·a^(k−1)·g` |
| $o=\tanh a$ | $\tanh a$ | `a.grad += (1 − o²)·g` |
| $o=e^a$ | $e^a$ | `a.grad += o·g` |
| $o=\mathrm{relu}(a)$ | $\max(0,a)$ | `a.grad += (a > 0)·g` |

**Çıkarma ve bölme** ayrıca yazılmaz: $a-b=a+(-1)\cdot b$, $a/b=a\cdot b^{-1}$. Python'un `__radd__`, `__rmul__` gibi yöntemleri `2 * x` gibi ifadelerin de çalışmasını sağlar.

### backward(): topolojik sıra ve ters gezinti {#backward}

:::tanım Topolojik sıralama
Yönlü döngüsüz bir grafikte düğümlerin, her kenar $u\to v$ için $u$'nun $v$'den önce geldiği bir sıralaması. Derinlik öncelikli arama ile: bir düğümü, **tüm girdileri listeye eklendikten sonra** listeye ekle.
:::

```python
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build(child)
                topo.append(v)            # girdilerinden sonra
        build(self)
        self.grad = 1.0                   # ∂L/∂L
        for v in reversed(topo):          # çıktıdan girdilere
            v._backward()
```

Neden **ters** topolojik sıra? Bir düğümün `_backward`'ı çağrıldığında, onun `grad`'ı **tamamlanmış** olmalı: onu kullanan tüm düğümler katkılarını eklemiş olmalı. Ters topolojik sıra tam olarak bunu garanti eder: bir düğüm, onu kullanan her düğümden sonra işlenir. Yanlış sırada gezersek (ör. düz sırada), bir düğüm gradyanını eksikken dağıtır ve sonuç sessizce yanlış olur.

Aşağıda bir ifade yaz: grafiği, topolojik sırayı, her düğümün değerini ve gradyanını gör; sonuç sayısal türevle karşılaştırılır.

::lab l1

:::örnek Karpathy'nin nöronu
$o=\tanh(x_1w_1+x_2w_2+b)$, $x_1=2$, $x_2=0$, $w_1=-3$, $w_2=1$, $b=6.8814$. İleri: $n=-6+0+6.8814=0.8814$, $o=\tanh(0.8814)=0.7071$. Geri: $\frac{\partial o}{\partial n}=1-o^2=0.5$; toplama gradyanı dağıtır: $\frac{\partial o}{\partial b}=0.5$, $\frac{\partial o}{\partial(x_1w_1)}=0.5$; çarpma takas eder: $\frac{\partial o}{\partial w_1}=x_1\cdot0.5=1$, $\frac{\partial o}{\partial x_1}=w_1\cdot0.5=-1.5$, $\frac{\partial o}{\partial w_2}=x_2\cdot0.5=0$ — girdisi sıfır olan ağırlık öğrenmez (M0.3). Laboratuvarda "nöron" senaryosunu seç ve doğrula.
:::

### Nöron, katman, MLP {#mlp}

Motor hazır olunca sinir ağı birkaç sınıftır:

```python
class Neuron:
    def __init__(self, nin):  self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]; self.b = Value(0)
    def __call__(self, x):    return sum((wi*xi for wi, xi in zip(self.w, x)), self.b).tanh()
    def parameters(self):     return self.w + [self.b]
class Layer:   # nout nöron, aynı girdiyi görür
class MLP:     # katmanlar art arda
```

Eğitim döngüsü M1.1'deki beş satırdır: ileri geçiş → kayıp → **gradyanları sıfırla** → `loss.backward()` → her parametre için `p.data -= lr * p.grad`. PyTorch'un `nn.Module`, `parameters()`, `optimizer.zero_grad()`, `optimizer.step()` yapıları bunların endüstriyel sürümüdür.

Aşağıda aynı tür bir MLP'yi (matris biçiminde, M3.2'nin denklemleriyle) iki ay verisinde eğit. Gizli katmanı kaldırınca lojistik regresyona döndüğünü, katman eklenince sınırın büküldüğünü gör; aynı hesabın skaler bir motorda kaç düğüm olacağına bak.

::lab l2

### Skaler motorun sınırı: neden tensörler? {#tensor}

micrograd her çarpma ve toplama için bir Python nesnesi oluşturur. 100 örneklik bir batch'te, 2–12–12–1'lik minik bir ağın tek ileri geçişi ~40 000 düğümdür; her biri bir Python nesnesi, bir kapanış, bir sözlük girişi. Python'da bir nesne oluşturmak ~100 ns, bir çarpma ise donanımda ~0.1 ns'nin altındadır: **zamanın neredeyse tamamı yönetim yüküdür**.

Tensör motoru (PyTorch, M3.3) aynı fikri büyük bloklara uygular: düğümler skaler değil tensördür, "çarpma" bir matris çarpımıdır, yerel türev bir Jacobian–vektör çarpımıdır (M0.3: $dA=GB^\top$). Aynı ağ birkaç düğüm ve birkaç optimize kernel çağrısıdır. Modülün **c** aşamasında bir nöronun ileri/geri geçişini NumPy ile tek satırlarda yazıp skaler motorla süresini karşılaştıracaksın.

| | micrograd | PyTorch |
|---|---|---|
| düğüm | skaler `Value` | `Tensor` (+ `grad_fn`) |
| yerel türev | Python kapanışı | C++/CUDA `backward` fonksiyonu |
| grafik | her ileri geçişte yeniden kurulur (dinamik) | aynı (define-by-run) |
| sıra | ters topolojik (DFS) | ters topolojik (öncelik kuyruğu, çok iş parçacıklı) |
| gradyan birikimi | `+=` | `.grad` birikir, `zero_grad()` |
| ara değerler | kapanışlarda | `ctx.save_for_backward` |

### Örnek başına gradyanlar {#ornek-basi}

`loss.backward()` kaybın ortalamasının (ya da toplamının) gradyanını verir: tüm örneklerin katkıları **toplanmıştır**. Bazen her örneğin kendi gradyanı gerekir:

- **Diferansiyel gizlilik** (DP-SGD): her örneğin gradyanını ayrı ayrı kırp, sonra gürültü ekle — tek bir örneğin modele etkisi sınırlanır.
- **Veri değerlendirme:** hangi örnekler modeli hangi yöne itiyor? (etki fonksiyonları, gradyan benzerliği)
- **Eğitim analizi:** gradyan gürültüsünün ölçümü (M1.2'deki $\Sigma/B$).

Naif yol: $n$ kez ayrı `backward` — $n$ kat yavaş. `torch.func` yolu: tek örnek için gradyan fonksiyonunu yaz, `vmap` ile batch boyutu üzerinde vektörleştir:

```python
from torch.func import grad, vmap, functional_call
def loss_one(params, x, y):  # tek örnek
    pred = functional_call(model, params, (x.unsqueeze(0),))
    return F.cross_entropy(pred, y.unsqueeze(0))
per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)   # her parametre: (n, *shape)
```

Modülün **e** aşaması bunu yapar ve sonucu döngüyle doğrular.

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| `Value` | data + grad + girdiler + `_backward` kapanışı |
| yerel türev | her işlem yalnız kendi türevini bilir; zincir kuralını motor uygular |
| `+=` | çok kullanılan değerlerin gradyanları toplanır; PyTorch'ta `.grad` birikir |
| sıra | ters topolojik: bir düğüm, onu kullanan her şeyden sonra işlenir |
| bellek | ileri geçişin ara değerleri geri geçiş için saklanır |
| ölçek | skaler motor eğitici ama yavaş; tensör motoru aynı fikir, büyük bloklar |

**Kaynaklar:** Karpathy, *micrograd* (MIT) ve "Neural Networks: Zero to Hero — 1: The spelled-out intro to neural networks and backpropagation"; Baydin ve ark. (2018), otomatik türev araştırması; Paszke ve ark. (2019), "PyTorch: an imperative style, high-performance deep learning library" (define-by-run autograd); PyTorch "Autograd mechanics" notu; `torch.func` belgeleri (per-sample gradients); Abadi ve ark. (2016), "Deep learning with differential privacy" (DP-SGD).
