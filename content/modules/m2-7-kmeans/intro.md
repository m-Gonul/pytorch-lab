$k$ merkez $c_1,\dots,c_k$ ve her noktanın atandığı küme $z_i$ için amaç [[inertia]]:

$$
J(C,z)=\sum_{i=1}^n\lVert x_i-c_{z_i}\rVert^2 .
$$

**Lloyd algoritması** iki adımı tekrarlar:

1. **Ata:** $z_i=\arg\min_k\lVert x_i-c_k\rVert^2$ (eşitlikte küçük $k$).
2. **Güncelle:** $c_k=\frac1{n_k}\sum_{i:z_i=k}x_i$. Küme boşsa ($n_k=0$) merkezi değiştirme.

Güncelleme adımının matris hâli: atama matrisi $A\in\{0,1\}^{n\times k}$ (one-hot) ile $C=\text{diag}(A^\top\mathbf 1)^{-1}A^\top X$. Atama adımı ise M2.1'deki mesafe matrisinin satır argmin'i.

Sonuç başlangıca bağlıdır. **k-means++**: ilk merkez rastgele; sonraki her merkez, noktaları en yakın mevcut merkeze kare uzaklığı $D(x)^2$ ile orantılı olasılıkla seçer.
