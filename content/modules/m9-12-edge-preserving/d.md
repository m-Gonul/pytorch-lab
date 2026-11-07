Türevlenebilir bilateral ve parametrelerini öğrenmek.

- `bilateral_t(img, r, sc, ss)` — `img (H, W)` float tensör (0–255), `sc`, `ss` tensör ya da sayı. `F.pad(…, mode="reflect")` (= `REFLECT_101`), `F.unfold` ile $(2r+1)^2$ komşu; dairesel maske $i^2+j^2\le r^2$; $w=e^{-(i^2+j^2)/2\sigma_s^2}e^{-(I_q-I_p)^2/2\sigma_r^2}$; $\sum wI_q/\sum w$.
- `fit_bilateral(noisy, clean, r=3, steps=150, lr=0.05)` — $\log\sigma_r$ ve $\log\sigma_s$'yi $(\log10,\log1)$'den başlatıp MSE ile Adam; `(sc, ss)` (float) döndür.

Kontrol, `bilateral_t`'nin yuvarlanmış çıkışını `cv2.bilateralFilter` ile karşılaştırır (`d = 7, 9`: OpenCV'nin bazı sürümleri `d ≤ 5`'i kırpan farklı bir koda yönlendirir) ve öğrenilen σ'ların ızgara aramasının en iyisine ulaştığını gösterir.
