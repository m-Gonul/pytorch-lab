Saf Python, liste listeleri. Gradyanlar **merkezi fark** (notlardaki $w_y=[-1,0,1]$, $w_x=w_y^T$): $f_x=f(y,x+1)-f(y,x-1)$, $f_y=f(y+1,x)-f(y-1,x)$.

- `harris_matrix(patch)` — $n\times n$ yamanın **iç** pikselleri ($1..n-2$) üzerinde kutu ağırlıklı $M$ → `[[a, b], [b, c]]`. Notlardaki iki 5 × 5 blokla aynı matrisleri vermeli.
- `harris_response(img, k, r=1)` — her piksel için $(2r+1)^2$ kutu pencerede $M$ ve $R$; gradyan ya da pencere görüntü dışına taşan piksellerde (kenardan $r+1$ içeride olmayanlar) 0.
- `local_maxima(R, thr, r)` — `R > thr` ve $(2r+1)^2$ komşuluğundaki (görüntü içinde kalan) her değerden büyük ya da eşit olan noktalar, `(x, y)` listesi, satır sırasıyla.
