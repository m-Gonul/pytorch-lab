micrograd her sayı için bir düğüm kuruyordu. Gerçek motorlar bir düğümü bir **tensör** yapar ve her işlem için tensör düzeyinde bir backward kuralı tanımlar. Gelen gradyan $G=\partial L/\partial Y$ ile:

| ileri | geri |
|---|---|
| $Y=A+B$ | $dA=\text{unb}(G)$, $dB=\text{unb}(G)$ |
| $Y=A\odot B$ | $dA=\text{unb}(G\odot B)$, $dB=\text{unb}(G\odot A)$ |
| $Y=AB$ | $dA=GB^\top$, $dB=A^\top G$ |
| $y=\sum_{\text{eksen}}A$ | $dA=G$ (toplanan eksende yayılmış) |
| $Y=f(A)$ eleman eleman | $dA=G\odot f'(A)$ |

**unb** = [[unbroadcast]]: ileri geçişte broadcasting bir girdiyi yaydıysa, gelen gradyanı o eksenler üzerinden toplayarak girdinin shape'ine geri indir. Kural: (1) baştaki fazla eksenler üzerinden topla, (2) girdide boyutu 1 olan eksenler üzerinden `keepdims=True` ile topla.

Reverse-mode (backprop), bir **vektör–Jacobian çarpımı** (VJP) hesaplar: $v^\top J$. Forward-mode ise **Jacobian–vektör çarpımı** (JVP): $Jv$. Tam Jacobian ya $m$ VJP ya $n$ JVP ister.
