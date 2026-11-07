Skaler motor her sayı için bir Python nesnesi oluşturur; $n$ örneklik bir batch için $n\cdot d$ düğüm. Tensör düzeyinde aynı nöronun ileri ve geri geçişi birkaç satır:

$$
o=\tanh(Xw+b)\in\mathbb{R}^{n},\qquad L=\sum_i o_i,\qquad
\nabla_w L=X^\top(1-o^2),\qquad \frac{\partial L}{\partial b}=\sum_i(1-o_i^2).
$$

- `neuron_fb(X, w, b)` → `(o, L, dw, db)`. **Döngü yok.** Bu, `Value` grafiğinde her örnek için ayrı ayrı yaptığın şeyin matris hâli: $X^\top$ çarpımı, örnekler üzerindeki gradyan toplamını tek işlemde yapıyor.
