$X^\top X$ ve $X^\top y$'yi hesapla, sonra iki $2\times2$ sistemi çöz. Ridge katsayıları OLS'ye göre **küçüldü**: $\lVert w\rVert^2$ 5'ten $2.66$'ya indi. $\lambda\to\infty$ iken tüm katsayılar sıfıra gider.

Soft-threshold'da üç durumu gör: büyük pozitif değer $t$ kadar küçülür, küçük değer **tam sıfır** olur, negatif değer işaretini koruyarak küçülür. Lasso'nun özellik seçmesinin tek mekanizması bu sıfırlama.
