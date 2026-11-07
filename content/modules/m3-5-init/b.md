Saf Python, skaler fonksiyonlar ve türevleri (`math` serbest):

- `relu`, `d_relu` (0'da türev 0 kabul et)
- `leaky_relu(x, a=0.01)`, `d_leaky_relu`
- `sigmoid`, `d_sigmoid` (kararlı sigmoid, M1.4)
- `tanh`, `d_tanh`
- `gelu(x)` — tanh yaklaşımı: $0.5x\big(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3))\big)$; `d_gelu`
- `silu(x)` $=x\,\sigma(x)$, `d_silu`

Kontrol her türevi sayısal türevle, GELU'yu da referans değerlerle karşılaştırır.
