Saf Python, forward-mode AD.

- `Dual(val, dot)` — `+`, `-`, `*` (sağdan sayılarla da: `__radd__`, `__rmul__`), ve modül fonksiyonları `dsin(d)`, `dexp(d)`.
- `jvp(f, x, v)` — `f`: `Dual` listesi → `Dual` listesi; `x`, `v` sayı listesi → `(f(x) değerleri, Jv)`.
- `jacobian_fwd(f, x)` — $n$ JVP ile ($v=e_j$), $m\times n$ liste.
