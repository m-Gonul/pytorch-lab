Gauss ağırlıklı Harris ve **tekrarlanabilirlik** (repeatability): görüntü bilinen bir dönüşümle değişince aynı köşelerin yeniden bulunma oranı.

- `harris_torch(img, sigma_d=1.0, sigma_i=2.0, k=0.04)` — `img (H, W)` float: (1) $\sigma_d$ Gauss ile yumuşat (ayrılabilir, yarıçap $\lceil3\sigma\rceil$, toplamı 1, `reflect` dolgu), (2) merkezi fark / 2 (`replicate`), (3) $f_x^2, f_xf_y, f_y^2$'yi $\sigma_i$ Gauss ile topla, (4) $R$.
- `detect(R, n=200, r=4, border=16)` — `max_pool2d` ile $(2r+1)^2$ yerel en büyükler, kenardan `border` piksel uzaktakiler arasından en büyük `n` tanesi; `(n, 2)` `[x, y]` float.
- `repeatability(img, angle, scale, adapt=False, tol=2.0)` — hazır `similarity` ile ikinci görüntüyü üret; ilkinde `harris_torch` + `detect`, ikincide `adapt` ise $\sigma_d\cdot s$, $\sigma_i\cdot s$ ve `r = int(4·s)`; ilk noktaları dönüşümle taşı, taşınanlardan ikinci görüntünün kenar bölgesine (16 piksel) düşmeyenler için en yakın tespite uzaklık `≤ tol` olanların oranı.
