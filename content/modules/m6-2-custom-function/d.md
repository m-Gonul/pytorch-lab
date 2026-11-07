`torch.autograd.Function` alt sınıfları:

- `LogSumExp` — son eksende; `forward` çıktıyı (ve gerekirse girdiyi) saklar; `backward` = `g[..., None] * exp(x - out[..., None])`.
- `RoundSTE` — ileri `torch.round`, geri özdeşlik.
- `GradReverse` — `apply(x, lam)`; ileri `x.view_as(x)`, geri `-lam * g` ve `lam` için `None`.

Kontrol `torch.autograd.gradcheck` ile ve $x=1000$ gibi büyük girdilerle sınar.
