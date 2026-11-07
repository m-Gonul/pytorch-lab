Düz ve residual derin ağda gradyanın katmanlar boyunca normunu **elle** geri yay ($D$ genişlik, $L$ katman, tanh):

$$
\text{düz: }h_{l+1}=\tanh(h_lW_l),\qquad \text{residual: }h_{l+1}=h_l+\tanh(h_lW_l),\qquad \mathcal L=\sum h_L .
$$

Geri: $g_L=\mathbf 1$; düz için $g_l=\big(g_{l+1}\odot(1-t^2)\big)W_l^\top$, residual için $g_l=g_{l+1}+\big(g_{l+1}\odot(1-t^2)\big)W_l^\top$ ($t$ o katmanın tanh çıktısı).

- `grad_norms(X, Ws, residual)` → uzunluğu $L+1$ liste: $\lVert g_0\rVert,\dots,\lVert g_L\rVert$ (Frobenius). Yalnız katman döngüleri (ileri ve geri, iç içe değil).

Kontrol 20 katmanda iki ağı karşılaştırır.
