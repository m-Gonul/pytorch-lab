Saf Python.

- `clip_histogram(hist, clip)` — OpenCV kuralı: eşik üstünü kes, fazlayı önce $\lfloor e/n\rfloor$ olarak tüm bölmelere, kalan $r$'yi $i=0$'dan başlayıp $\max(\lfloor n/r\rfloor,1)$ adımla birer birer dağıt ($n$ = bölme sayısı). Yeni liste.
- `tile_lut(hist, clip_limit, total)` — `clip = max(int(clip_limit · total / n), 1)` (clip_limit ≤ 0 ise kırpma yok), `clip_histogram`, LUT[$i$] = `round(cdf[i] · (n − 1) / total)`.
- `bilateral(img, d, sigma_color, sigma_space)` — OpenCV tanımı: `radius = d // 2` (d ≤ 0 ise `round(1.5·sigma_space)`), en az 1; yalnız $\sqrt{i^2+j^2}\le$ radius olan ofsetler; `BORDER_REFLECT_101`; $w=e^{-r^2/2\sigma_s^2}e^{-\Delta^2/2\sigma_c^2}$; çıkış `round(Σwv/Σw)`.
