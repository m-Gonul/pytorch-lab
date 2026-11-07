Morfoloji = toplama yerine max/min alan "konvolüsyon".

- `dilate_t(f, w)` — `f (B, 1, H, W)` float, `w (k, k)`: düz olmayan **ders kitabı** genişlemesi $\max_{b}[f(z-b)+w(b)]$. `F.unfold` ile; dışarısı $-10^4$. Düz SE: `w`'de eleman 0, eleman olmayan $-\infty$ (pratikte çok negatif).
- `learn_se(img, target_se, steps=1500, lr=0.02, patch=128, seed=0)` — sıfırdan başlayan 3 × 3 `w`'yu, rastgele yamalarda `dilate_t(f, w)` ile `cv2.dilate(img, target_se)` arasındaki MSE ile Adam'la öğren; `w`'yu döndür.
- `tophat_t(f, k)` — kare $k\times k$ düz SE ile top-hat; erozyon `-max_pool2d(-f)`, genişleme `max_pool2d` (dolgu $-\infty$ olduğundan OpenCV kenar kuralıyla aynı).

Gradyan yalnız `max`'ı kazanan elemana akar: öğrenme eleman olmayan köşeleri "asla kazanamayacakları" kadar aşağı iter.
