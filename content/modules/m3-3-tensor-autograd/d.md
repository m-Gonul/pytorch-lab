Şimdi PyTorch'ta, ama **autograd'sız**: gradyanları tensör işlemleriyle elle yaz ve autograd'ın bulduğuyla karşılaştır.

$$
Z=(a\odot b+a)\,W,\qquad L=\sum\text{ReLU}(Z),\qquad a:(n,d),\ b:(d,),\ W:(d,k).
$$

- `unbroadcast_torch(G, shape)` — genel kural, `sum_to_size` kullanmadan.
- `manual_grads(a, b, W)` → `(da, db, dW)`; `backward`, `torch.autograd` ya da `torch.func` **yasak**.
