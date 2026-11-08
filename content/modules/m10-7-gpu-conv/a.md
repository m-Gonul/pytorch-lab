<!-- çözüm -->

K = 5: $(16+4)^2=400$, $400/256=1.5625$ okuma/çıktı, $25/1.5625=16$. K = 21: $36^2/256=5.06$, $441/5.06=87.1$.

Matmul fazı: $2T^3$ FLOP, A ve B karoları $2T^2\cdot4$ bayt → $T/4$. im2col: her piksel 441 kez yazılır: $512^2\cdot441\cdot4$ B = 441 MiB (görüntünün kendisi 0.25 MiB uint8). Sabit bellek: 16 384 float → 128 × 128.
