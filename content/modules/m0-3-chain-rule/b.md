İki şey yaz:

- `f_backward(x, y)` — ileri geçişi ara değişkenlerle (`a = x + y`, `out = a * y`) yap, sonra geri geçişi **elle**, düğüm düğüm yaz. `(out, dfdx, dfdy)` döndür. Formülü doğrudan yazma ($\partial f/\partial y = x+2y$); grafiği takip et — M3.1'de bunu otomatikleştireceksin.
- `num_grad(f, xs, h=1e-5)` — herhangi bir `f(*xs)` için merkezi farkla kısmi türev listesi. `xs`'i değiştirme; her koordinat için bir kopya oluştur (`if`/`for` serbest).

`numpy` ve `torch` yasak.
