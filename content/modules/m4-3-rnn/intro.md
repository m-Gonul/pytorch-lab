Bir RNN her zaman adımında aynı ağırlıklarla bir gizli durumu günceller:

$$
h_t=\tanh\big(x_tW_x+h_{t-1}W_h+b\big),\qquad t=1..T .
$$

Zamanda açıldığında bu, $T$ katmanlı ve **ağırlıkları paylaşılan** bir ileri beslemeli ağdır. Backpropagation through time (BPTT) bu açılmış grafikte sıradan backprop'tur:

$$
\frac{\partial L}{\partial h_{t-1}}=\Big(\frac{\partial L}{\partial h_t}\odot(1-h_t^2)\Big)W_h^\top+\frac{\partial L_{t-1}}{\partial h_{t-1}},\qquad
\frac{\partial L}{\partial W_h}=\sum_t h_{t-1}^\top\Big(\frac{\partial L}{\partial h_t}\odot(1-h_t^2)\Big).
$$

Uzak geçmişe giden gradyan, $\prod_t W_h^\top\text{diag}(1-h_t^2)$ çarpımını içerir. Bu çarpım büyük $T$'de sıfıra (sönme) ya da sonsuza (patlama) gider. Patlamaya çare gradient clipping; sönmeye çare kapılı hücreler (M4.4).

Tanh olmasaydı rekürsiyon doğrusal olurdu: $h_t=a_t\odot h_{t-1}+b_t$. Bunun özel bir yanı var: adımlar birleşmeli bir işlemle birleşir ve [[associative scan]] ile $O(\log T)$ paralel adımda hesaplanır. Mamba gibi state-space modellerin hızı buradan gelir.
