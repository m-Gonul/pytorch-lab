Saf Python (`cmath` serbest), karmaşık sayı listeleri.

- `dft(x)` — tanımdan, $O(N^2)$.
- `fft(x)` — özyinelemeli Cooley-Tukey: $X_k=E_k+e^{-j2\pi k/N}O_k$, $X_{k+N/2}=E_k-e^{-j2\pi k/N}O_k$; $N$ 2'nin kuvveti değilse `dft`'ye düş.
- `idft(X)` — $\frac1N\sum X_ue^{+j2\pi ux/N}$ (ipucu: eşlenik, `fft`, eşlenik, $/N$).
- `dft2(img)` — ayrılabilirlik: her satıra, sonra her sütuna `fft`.
