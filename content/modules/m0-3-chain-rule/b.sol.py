def f_backward(x, y):
    """f = (x + y) * y. (out, df/dx, df/dy) döndür — geri geçişi düğüm düğüm yaz."""
    # ileri
    a = x + y
    out = a * y
    # geri: dout = 1 ile başla
    dout = 1.0
    da = dout * y        # çarpım düğümü: ∂(a·y)/∂a = y
    dy = dout * a        # çarpım düğümü: ∂(a·y)/∂y = a
    dx = da * 1.0        # toplama düğümü gradyanı aynen geçirir
    dy += da * 1.0       # y'nin ikinci yolu: katkılar toplanır
    return out, dx, dy


def num_grad(f, xs, h=1e-5):
    """f(*xs)'in her koordinata göre merkezi fark türevi (liste)."""
    g = []
    for i in range(len(xs)):
        up, dn = list(xs), list(xs)
        up[i] += h
        dn[i] -= h
        g.append((f(*up) - f(*dn)) / (2 * h))
    return g


print(f_backward(2.0, 3.0))                       # (15.0, 3.0, 8.0)
print(num_grad(lambda x, y: (x + y) * y, [2.0, 3.0]))
