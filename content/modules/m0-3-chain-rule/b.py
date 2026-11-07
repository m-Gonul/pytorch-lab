def f_backward(x, y):
    """f = (x + y) * y. (out, df/dx, df/dy) döndür — geri geçişi düğüm düğüm yaz."""
    # ileri
    a = x + y
    out = a * y
    # geri: dout = 1 ile başla
    dout = 1.0
    # da = ?        (çarpım düğümünden a'ya)
    # dy = ?        (çarpım düğümünden y'ye, doğrudan)
    # dx = ?        (toplama düğümünden x'e)
    # dy += ?       (toplama düğümünden y'ye — ikinci yol!)
    raise NotImplementedError


def num_grad(f, xs, h=1e-5):
    """f(*xs)'in her koordinata göre merkezi fark türevi (liste)."""
    ...


print(f_backward(2.0, 3.0))                       # (15.0, 3.0, 8.0)
print(num_grad(lambda x, y: (x + y) * y, [2.0, 3.0]))
