Şimdi grafiği [[autograd]] kursun. PyTorch her işlemi kaydeder, `.backward()` grafiği ters [[topolojik sıralama]] ile gezer ve `x.grad`'a yazar; bir değişken iki yoldan kullanıldıysa katkıları kendisi toplar.

- `grads_autograd(x, y)` — iki skaleri `requires_grad=True` tensöre çevir, $f$'yi hesapla, `backward()` çağır, `(dfdx, dfdy)` döndür.
- `jacobian_torch(p)` — `p = tensor([x, y])` için $g(p)=[\,xy,\;x+y^2\,]$'nin Jacobian'ı. `torch.autograd.functional.jacobian` ya da `torch.func.jacrev` kullan.
