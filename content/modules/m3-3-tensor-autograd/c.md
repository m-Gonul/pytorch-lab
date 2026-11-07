NumPy üzerinde bir autograd motoru. `Tensor(data)`: `self.data` (float64 `ndarray`), `self.grad` (aynı shape, sıfırla başlar), `self._prev`, `self._backward`.

Yazılacaklar (hepsi tensör düzeyinde, **eleman döngüsü yok**):

- `unbroadcast(grad, shape)` — b'deki kuralın genel hâli (`grad.sum(axis=…, keepdims=…)`).
- `__add__`, `__mul__`, `__matmul__`, `__neg__`, `__sub__`, `__pow__(k)` (sayı), `__truediv__`, `__radd__`, `__rmul__`.
- `sum(axis=None, keepdims=False)`, `mean()`, `relu()`, `exp()`, `log()`.
- `backward()` — topolojik sıra, `self.grad = np.ones_like(self.data)`, ters sırada `_backward`.

Sayı ile işlemler (`t * 2`, `1 - t`) sayıyı `Tensor`'a sarsın. Kontrol: rastgele ifadeler sayısal gradyanla, `x * x` birikimi, ve motorunla eğitilen softmax regression.
