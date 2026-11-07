Bir sinir ağında zamanın büyük kısmı tek bir işlemde geçer: matris çarpımı. Bir [[lineer katman]] $y = xW^\top + b$, bir [[attention]] skoru $QK^\top$, bir [[convolution]] (im2col sonrası) — hepsi aynı çekirdeğe iner.

$$
C = AB,\qquad C_{ij} = \sum_{t=1}^{k} A_{it}\,B_{tj},\qquad (n\times k)(k\times m) \to (n\times m)
$$

Her $C_{ij}$, $A$'nın $i$. satırı ile $B$'nin $j$. sütununun iç çarpımıdır (M0.1). Toplam iş $n\cdot k\cdot m$ çarpma ve yaklaşık o kadar toplamadır; $n=k=m$ için $O(n^3)$.

İkinci kavram [[broadcasting]]: shape'leri farklı iki dizi arasında eleman eleman işlem yaparken, küçük olanın eksik boyutlarının "kopyalanmış gibi" davranması. Kural: boyutlar **sağdan** hizalanır; her eksende ya eşittirler ya biri 1'dir (ya da yoktur). Kopya bellekte yapılmaz.
