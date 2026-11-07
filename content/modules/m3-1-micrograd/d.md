Şimdi aynı iki örneği PyTorch'a yaptır ve elle bulduklarınla karşılaştır.

- `karpathy_grads()` — $a=2$, $b=-3$, $c=10$, $f=-2$ için `requires_grad=True` skaler tensörlerle $L=(a\cdot b+c)\cdot f$; `{"a": …, "b": …, "c": …, "f": …}` sözlüğüyle gradyanları döndür.
- `neuron_torch(x, w, b)` → `(o, dw, dx)` — $o=\tanh(x\cdot w+b)$'nin $w$ ve $x$'e göre gradyanları. (Girdiye göre gradyan, adversarial örnekler ve saliency haritalarının temeli.)

Kontrol ayrıca `o.grad_fn` zincirini gezerek PyTorch'un grafiğinin seninkine benzediğini gösterir.
