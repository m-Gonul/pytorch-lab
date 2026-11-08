NumPy.

- `sqrtm_psd(A)` — simetrik yarı pozitif tanımlı matrisin karekökü (`eigh`, negatif özdeğerleri 0'a kırp). Döngüsüz.
- `sqrtm_ns(A, iters=50)` — Newton–Schulz yinelemesi: $Y_0=A/\|A\|_F$, $Z_0=I$, $T=\tfrac12(3I-Z_kY_k)$, $Y_{k+1}=Y_kT$, $Z_{k+1}=TZ_k$; sonuç $Y\sqrt{\|A\|_F}$ (tek döngü).
- `frechet(mu1, S1, mu2, S2)` — iz terimini simetrik biçimle: $\mathrm{tr}\sqrt{S_1^{1/2}S_2S_1^{1/2}}$.
- `fid(F1, F2)` — öznitelik matrislerinden (satır = örnek; kovaryans `ddof=1`).
- `inception_score(P, splits=1)` — `P (N, C)`; batch'leri `splits` parçaya bölüp skorların ortalaması ve std'si.
