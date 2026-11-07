NumPy, **döngü yok** (çekirdek boyutu üzerinde de).

- `box_separable(img, size)` — sıfır doldurma; önce yatay, sonra dikey 1-boyutlu ortalama. Kayan toplamı birikimli toplamla al: pencere toplamı `cs[i+size] - cs[i]`.
- `integral(img)` → `(H+1, W+1)` `int64`, ilk satır/sütun sıfır (`cv2.integral` gibi).
- `box_integral(img, size)` — görüntüyü `r = size//2` sıfırla doldur, integralini al, her çıkış için **4 arama + 1 bölme**.
- `gaussian_kernel(ksize, sigma=0)` → `(ksize,)`, toplamı 1; `sigma ≤ 0` ise OpenCV kuralı. **İnce ayrıntı:** OpenCV `sigma ≤ 0` ve `ksize ∈ {1, 3, 5, 7}` için formülü değil, binom katsayılarına yakın sabit tabloları kullanır: 3 → $[1,2,1]/4$, 5 → $[1,4,6,4,1]/16$, 7 → $[1,3.5,7,9,7,3.5,1]/32$. Bunları da destekle (`cv2.getGaussianKernel(3, 0)` ile kendin gör).
- `gaussian_blur(img, ksize, sigma=0)` — ayrılabilir, `REFLECT_101` doldurma (`np.pad(…, mode="reflect")`).
