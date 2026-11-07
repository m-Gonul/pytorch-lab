**İç içe geçme doğrulayıcı** (saf Python, model denetimi). Bir thread komut listesidir:

- `("load", var, reg)`, `("add", reg, k)`, `("store", var, reg)` — yazmaçlar thread'e özel;
- `("lock", m)`, `("unlock", m)` — kilit başkasındaysa thread ilerleyemez.

- `explore(threads, init)` → `(sonuçlar, kilitlenme_var_mı)`: tüm erişilebilir durumları DFS ile gez (durum = program sayaçları, yazmaçlar, paylaşılan değişkenler, kilit sahipleri; ziyaret edilenleri hatırla). Bütün thread'ler bittiğinde paylaşılan değişkenlerin son hâlini (sıralı `tuple` öğeleri) kümeye ekle; hiçbir thread ilerleyemiyor ama hepsi bitmemişse kilitlenme.
- `count_interleavings(lengths)` — kilitsiz programların toplam sıralama sayısı (multinom katsayısı).
