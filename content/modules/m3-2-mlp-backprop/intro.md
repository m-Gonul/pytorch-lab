İki katmanlı bir MLP (batch hâli, satırlar örnek):

$$
Z_1=XW_1+b_1,\quad H=\text{ReLU}(Z_1),\quad S=HW_2+b_2,\quad L=\text{CE}(S,y).
$$

Geri geçiş her katman için yerel bir kural; M3.1'deki skaler kuralların matris hâli:

| katman | ileri | geri ($dY$ gelen gradyan) |
|---|---|---|
| affine | $Y=XW+b$ | $dX=dY\,W^\top$, $dW=X^\top dY$, $db=\sum_i dY_i$ |
| ReLU | $Y=\max(0,X)$ | $dX=dY\odot[X>0]$ |
| softmax + CE | $L=\frac1n\sum-\log p_{y_i}$ | $dS=\frac1n(P-Y_{\text{onehot}})$ |

CS231n'in tasarımı: her katman `forward(x) → (out, cache)` ve `backward(dout, cache) → gradyanlar`. `cache`, geri geçişte gereken ileri değerleri tutar. Ağın geri geçişi, bu fonksiyonları ters sırada çağırmaktan ibaret.
