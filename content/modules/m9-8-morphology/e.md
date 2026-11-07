**van Herk / Gil-Werman:** uzunluğu $k$ olan pencerede max, eleman başına **3 karşılaştırma** ile, $k$'dan bağımsız. Diziyi $k$'lık bloklara böl; her blokta soldan kümülatif max $g$ ve sağdan kümülatif max $h$. $[i,i+k-1]$ penceresi en fazla iki bloğa yayılır: $\max(h_i, g_{i+k-1})$.

- `running_max(x, k)` — `x (..., L)` float, merkezli tek $k$, dışarısı $-\infty$; son eksen boyunca. `torch.cummax`, `flip`, `reshape` ile, döngüsüz.
- `rect_dilate(img, kh, kw)`, `rect_erode(img, kh, kw)` — `uint8` tensör, `cv2.dilate/erode(img, np.ones((kh, kw)))` ile birebir (ayrılabilirlik: önce satırlar, sonra sütunlar). Erozyon için $\min(x)=-\max(-x)$.
