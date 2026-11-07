`gradient_penalty(D, real, fake, alpha)` — `alpha: (B, 1)`.

1. $\hat x=\alpha\,x+(1-\alpha)\,\tilde x$, `requires_grad_(True)`.
2. `g, = torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)` — örnekler bağımsız olduğundan toplamın gradyanı = örnek başı gradyanlar.
3. $\operatorname{mean}_i(\lVert g_i\rVert_2-1)^2$.

**Döngü yok.** `create_graph=True` olmazsa ceza ayırıcının parametrelerine gradyan vermez; kontrol bunu da sınar.
