Büyük bir stok MDP'si, vektörel.

- `inventory_mdp(C=50, max_order=20, lam=8.0, price=10.0, cost=4.0, hold=0.5, short=2.0)` → `(P, R)`: durum stok $0..C$, eylem sipariş $0..$`max_order` (stoğu $C$'de kırp), talep $D\sim$ Poisson($\lambda$) (kuyruğu $C+$`max_order`'da birikir). Satış $\min(s+a,D)$; ödül $=$ fiyat·beklenen satış − maliyet·$a$ − tutma·beklenen kalan − kıtlık·beklenen karşılanmayan. **Döngüsüz** (`scipy.stats.poisson.pmf`, yayın).
- `vi_vectorized(P, R, gamma, eps)` ve karşılaştırma için durum döngülü `vi_loop` (verildi).

Kontrol, en iyi politikanın "(s, S) politikası" yapısını (stok bir eşiğin altındayken hedef düzeye tamamla) gösterir.
