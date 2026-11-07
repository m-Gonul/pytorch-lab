Saf Python, **tek örnek**, vektörler liste, matrisler satır listesi ($W$: `out × in`, yani $z=Wx+b$).

- `linear_forward(x, W, b)` → `z`
- `linear_backward(dz, x, W)` → `(dx, dW, db)`
- `relu_forward(z)`, `relu_backward(dh, z)`
- `mlp_loss(x, y, W1, b1, w2, b2)` → `(L, grads)`; $\hat y=w_2\cdot\text{ReLU}(W_1x+b_1)+b_2$, $L=\frac12(\hat y-y)^2$; `grads` sözlüğü: `"W1", "b1", "w2", "b2", "x"`. Önceki fonksiyonları kullan.
