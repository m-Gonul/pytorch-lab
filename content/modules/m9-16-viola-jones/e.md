Kaskad ve vektörel tarama. Hazır verilenler: d'deki eğitim (`train`), öznitelik tablosu.

- `make_cascade(F_pos, H, sizes=(2, 10, 30))` — aşama $i$, ilk `sizes[i]` zayıf sınıflandırıcıdır; eşiği eğitim yüzlerinin **hepsini** geçirecek şekilde `min(score)` (ve $\frac12\sum\alpha$'yı aşmayacak şekilde) seçilir. `[(k, thr), …]`.
- `scan(img, R, S, A, H, cascade, win=25)` → `(boxes, counts)`: tüm `(y, x)` pencereleri (adım 1); her pencere için ortalama/std (integral ve kare integral), öznitelik değerleri `(Nw, K)` gelişmiş indekslemeyle; **aşama aşama** yalnız hayatta kalan pencerelerle devam; `counts` her aşamaya giren pencere sayısı; `boxes` son aşamayı geçen `(x, y, score)` (score: son aşamanın skoru). Aşama döngüsü serbest.
- `nms(boxes, r=12)` — skora göre azalan sırada, kabul edilmiş bir kutuya `r`'den yakın merkezleri at.
