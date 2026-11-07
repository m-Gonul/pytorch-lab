En iyi $k$'yı seçmek için birçok $k$ dener. Her $k$ için yeniden sıralayıp oy saymak gereksiz: sıralama bir kez yapılır, oylar **kümülatif** toplanır.

$$
V_{i,m,c}=\sum_{r=1}^{m}[\,y_{\text{nb}(i,r)}=c\,]\quad\Rightarrow\quad \hat y_i^{(k)}=\arg\max_c V_{i,k,c}.
$$

`one_hot` ile `(n_te, k_max, C)` tensörü kur, `cumsum(dim=1)` al; $k$ için gereken satır `V[:, k-1, :]`. Tüm $k$'lar tek indekslemeyle: `V[:, ks - 1, :]`.

- `predict_many_k(D, ytr, ks, C)` → `(len(ks), n_te)`. **Döngü yok.** Sıralama için `torch.sort(D, dim=1, stable=True)` kullan (b–c'deki eşitlik kuralı).

Kontrol, sonucu her $k$ için ayrı tahminle karşılaştırır ve 20 $k$ değeri için hız tablosu üretir; ayrıca mesafe hesabının üç yolunu yarıştırır.
