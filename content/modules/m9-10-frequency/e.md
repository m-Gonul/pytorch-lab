`conv2d_fft(x, w, padding=0)` — `F.conv2d(x, w, padding=padding)` ile aynı (yani **çapraz korelasyon**), `x (B, Cin, H, W)`, `w (Cout, Cin, kh, kw)`, `groups=1`, `stride=1`.

1. `x`'i `padding` kadar sıfırla doldur → $H_p\times W_p$.
2. Boyut $(H_p+k_h-1, W_p+k_w-1)$'de `rfft2(x)` ve `rfft2(w.flip(-2, -1))` (korelasyon = ters çevrilmiş çekirdekle konvolüsyon).
3. Kanallar üzerinden topla: `einsum("bihw,oihw->bohw")`.
4. `irfft2`, geçerli bölge: `[kh-1 : kh-1+Hout, kw-1 : kw-1+Wout]`.

Karmaşıklık: doğrudan $O(HWk^2)$, FFT $O(HW\log HW)$ — kesişim noktasını ölç.
