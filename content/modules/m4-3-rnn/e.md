Doğrusal rekürsiyon $h_t=a_t\odot h_{t-1}+b_t$ ($h_0=0$), `a, b: (T, …)`.

Her adım bir afin dönüşüm $h\mapsto a\,h+b$; iki adımın bileşimi yine afin: $(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\;a_2b_1+b_2)$. **Hillis–Steele** scan: `offset = 1, 2, 4, …` için her $t\ge\text{offset}$'te

```text
B[t] ← A[t]·B[t−offset] + B[t]
A[t] ← A[t]·A[t−offset]
```

(sağ taraflar güncellemeden **önceki** değerler). $\lceil\log_2T\rceil$ turdan sonra $B[t]=h_t$.

- `linear_scan(a, b)` → `h` (aynı shape). Yalnız `offset` döngüsü ($\log T$ tur); zaman üzerinde döngü yok.
