[[k-nn|k-en yakın komşu]] öğrenmez, ezberler: tahmin zamanında sorgu noktasına en yakın $k$ eğitim örneğini bulur ve çoğunluk oyuna bakar. Algoritmanın tüm maliyeti bir [[mesafe matrisi]]nde:

$$
D_{ij}=\lVert x^{\text{test}}_i-x^{\text{train}}_j\rVert^2,\qquad D\in\mathbb{R}^{n_{te}\times n_{tr}}.
$$

Bu modül CS231n'in ilk ödevinin izinden gidiyor: aynı $D$'yi iki döngüyle, tek döngüyle ve hiç döngüsüz yazmak. Döngüsüz hâlin anahtarı kareyi açmak:

$$
\lVert a-b\rVert^2=\lVert a\rVert^2+\lVert b\rVert^2-2\,a\cdot b
\quad\Longrightarrow\quad
D = \underbrace{s_{te}}_{(n_{te},1)} + \underbrace{s_{tr}^\top}_{(1,n_{tr})} - 2\,X_{te}X_{tr}^\top .
$$

Son terim tek bir matris çarpımı (M0.2); ilk ikisi [[broadcasting]] ile $(n_{te}\times n_{tr})$'ye yayılır.

**Eşitlik kuralları** (bütün aşamalarda aynı): mesafesi eşit komşulardan indeksi küçük olan önce gelir; oylar eşitse küçük sınıf etiketi kazanır.
