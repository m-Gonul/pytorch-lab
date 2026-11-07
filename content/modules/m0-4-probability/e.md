**Bootstrap**: veriden yerine koyarak $n$ örnek çek, istatistiği hesapla, bunu $B$ kez tekrarla. $B$ istatistiğin dağılımı, tahminin belirsizliğini verir: ortalamanın standart hatası teoride $\sigma/\sqrt n$.

Döngüyle $B$ kez çekmek yerine tüm indisleri bir kerede çek: `idx = torch.randint(n, (B, n))`, sonra `x[idx]` `(B, n)` ve satır ortalaması.

- `bootstrap_means(x, B, generator)` → `(B,)`. **Döngü yok.** Rastgelelik için verilen `generator`'ı kullan.
- `batched_gaussian_mle(X)` — `X: (B, n)`, her satır ayrı bir veri seti; `(mu: (B,), var: (B,))`. **Döngü yok.**

Kontrol, bootstrap standart sapmasını teorik $\hat\sigma/\sqrt n$ ile karşılaştırır ve döngülü sürümle hız tablosu üretir.
