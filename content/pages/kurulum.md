# Motorlar ve kurulum

Sitedeki her kod kutusu üç motordan birinde çalışır. Hangi aşamanın nerede çalıştığı kutunun sol üstünde yazar; bir aşamanın kontrolleri her motorda **aynı** `labcheck.py` dosyasıyla yapılır, yani tarayıcıda geçen kod yerelde ve Colab'da da geçer.

| Aşama | Varsayılan motor | Neden |
|---|---|---|
| 1 Elle | motor yok | Değerleri sen hesaplarsın, sayfa anında kontrol eder |
| 2 Döngü, 3 NumPy | Tarayıcı ([[pyodide|Pyodide]]) | Saf Python + NumPy/pandas tarayıcıda tam destekli |
| 4 PyTorch, 5 Vektörize | Yerel PyTorch | Pyodide'de PyTorch yok; hız ölçümü de gerçek donanımda anlamlı |

## Tarayıcı motoru

Kurulum yok. İlk "Kontrol et"te Python (WebAssembly'ye derlenmiş CPython) jsDelivr'dan iner, sonra tarayıcı önbelleğinde kalır. Kod bir [[web worker]] içinde çalışır; sonsuz döngüye girersen kutudaki **Durdur** worker'ı öldürüp yeniden başlatır. Kodun ve ilerlemen yalnızca bu tarayıcıda (`localStorage`) tutulur.

## Yerel PyTorch motoru {#motorlar}

Bir kez [uv](https://docs.astral.sh/uv/getting-started/installation/) kur (tek dosyalık Python paket/ortam yöneticisi), sonra:

```bash
uv run https://m-gonul.github.io/pytorch-lab/runner/torchlab.py
```

`uv` betiğin başındaki bağımlılık listesini (PEP 723) okur, torch/numpy/pandas'lı izole bir ortam kurar ve önbelleğe alır; ikinci açılış saniyeler sürer. Motor terminale bir bağlantı yazar ve tarayıcıda açar; o bağlantı bu siteyi motora **token** ile bağlar.

<div id="runner-connect" class="runner-connect"></div>

Güvenlik modeli: motor yalnızca `127.0.0.1`'i dinler, her istekte oturum token'ını ister ve yalnızca bu sitenin origin'inden gelen isteklere CORS izni verir. Kod zaten senin kodun ve senin makinende çalışır.

Seçenekler:

```bash
# Linux'ta CUDA paketlerini indirmeden, yalnız CPU torch:
uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  https://m-gonul.github.io/pytorch-lab/runner/torchlab.py

# farklı port, daha uzun zaman aşımı, siteyi kendi bilgisayarında geliştiriyorsan:
uv run runner/torchlab.py --port 9000 --timeout 600 --site http://localhost:8000/
```

Safari, `https` bir sayfadan `http://127.0.0.1` adresine isteği engelleyebilir; Chrome, Edge ve Firefox'ta sorun yok.

### Faz 6–7 için ek araçlar

- **C++ derleyicisi** (M7.1 `load_inline`, M6.5 `torch.compile` CPU arka ucu): Linux'ta `g++`, macOS'ta Xcode Command Line Tools (`xcode-select --install`), Windows'ta Visual Studio Build Tools. `ninja` motorun ortamına otomatik kurulur.
- **Triton** (M7.2): yalnız Linux x86-64'te paket olarak var; motor orada otomatik kurar. GPU yoksa çekirdekler `TRITON_INTERPRET=1` yorumlayıcı modunda CPU'da çalışır (yavaş ama doğru). macOS ve Windows'ta M7.2'nin PyTorch aşamaları için Colab'ı kullan.
- **Çok süreçli işler** (M7.4 DDP/FSDP): CPU'da `gloo` arka ucuyla iki süreç başlatılır; ek kurulum gerekmez.

## Colab

Her modül sayfasının başında **Colab'da aç** bağlantısı var. Notebook'ta her aşama bir `%%writefile` hücresi (senin kodun) ve bir kontrol hücresidir; kontroller aynı `labcheck.py` ile çalışır. GPU gerektiren Faz 7 modülleri için Colab'da *Runtime → Change runtime type → GPU* seç.

## Deneysel: tarayıcıda PyTorch

PyTorch'u WebAssembly'ye derleyip JupyterLite'ta çalıştıran çalışma (emscripten-forge) henüz deneysel ve resmi olarak yayımlanmadı. Kararlı hâle geldiğinde 4. ve 5. aşamalar için üçüncü motor olarak eklenecek.
