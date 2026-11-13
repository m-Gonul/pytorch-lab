Özellik sayısı örnek sayısına yaklaşınca ya da özellikler birbiriyle ilişkiliyken en küçük kareler büyük, kararsız katsayılar üretir: [[aşırı uyum]]. Çare, katsayıların büyüklüğünü kayba eklemek:

$$
\text{Ridge: }\ \min_w \lVert y-Xw\rVert^2+\lambda\lVert w\rVert_2^2,\qquad
\text{Lasso: }\ \min_w \frac1{2n}\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1 .
$$

Ridge'in kapalı formu var: gradyanı sıfıra eşitle,

$$
(X^\top X+\lambda I)\,w=X^\top y .
$$

$\lambda I$ köşegeni büyütür; sistem her zaman terslenebilir olur. Lasso'nun kapalı formu yok ama tek bir koordinatta çözüm basit: diğerleri sabitken $w_j$ için optimum bir [[soft-threshold]]:

$$
w_j \leftarrow \frac{S\!\left(\tfrac1n x_j^\top r_j,\ \lambda\right)}{\tfrac1n\lVert x_j\rVert^2},\qquad r_j=y-\sum_{k\ne j}x_kw_k .
$$

Bu adımı koordinatlar üzerinde tekrarlamak **coordinate descent**; glmnet ve scikit-learn'ün `Lasso`'su bunu yapar. Bias (intercept) cezalandırılmaz: veriyi ortalayıp sonra $b=\bar y-\bar x^\top w$ ile geri koyarız.

İki cezayı birleştiren **elastic net** ($\lambda_1\lVert w\rVert_1+\lambda_2\lVert w\rVert_2^2$) L1'in seyrekliğiyle L2'nin kararlılığını taşır ve ilişkili öznitelikleri birlikte seçer; tek koordinatta çözümü lasso adımının hemen üzerine oturur (aşağıda türetiyoruz). λ değerini elle tutmak yerine k-fold çapraz doğrulama ile seçeriz — [[cross-validation|cross-validation]] bölümünde 1-SE kuralına kadar gidiyoruz.
