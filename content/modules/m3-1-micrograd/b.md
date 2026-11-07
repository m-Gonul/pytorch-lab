`Value` sınıfını tamamla. `__init__`, `__repr__` ve `__add__` hazır; `__add__`'ın `_backward`'ını şablon olarak oku.

Yazılacaklar:

- `__mul__` — `other` sayıysa önce `Value`'ya sar.
- `__pow__(k)` — `k` yalnız `int`/`float`.
- `tanh()` — `math.tanh` serbest.
- `backward()` — derinlik öncelikli arama ile topolojik sıra kur (ziyaret edilenleri bir `set`'te tut), `self.grad = 1.0`, sonra ters sırada `_backward()` çağır.
- Kolaylıklar: `__radd__`, `__rmul__`, `__neg__`, `__sub__`, `__rsub__`, `__truediv__` (son ikisi `__pow__(-1)` ile).

Kontrol: elle örnek, tanh nöronu, `a + a` (gradyan birikmesi), uzun bir ifadenin [[sayısal türev]] ile karşılaştırılması.
