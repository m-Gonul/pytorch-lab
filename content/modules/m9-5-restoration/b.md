Saf Python; `uint8` değerli liste listesi; kenarlar **kopyalama** (`BORDER_REPLICATE`, `cv2.medianBlur`'un kuralı).

- `median_filter(img, k)` — `k × k` (tek).
- `cw_median(img, k, w)` — merkez `w` kez sayılır (pencereye `w-1` kopya ekle).
- `adaptive_median(img, smax=7)` — pencere 3'ten başlar, 2'şer büyür.

Kontrol `cv2.medianBlur` ile birebir ve uyarlamalı median'ı başvuru uygulamasıyla karşılaştırır.
