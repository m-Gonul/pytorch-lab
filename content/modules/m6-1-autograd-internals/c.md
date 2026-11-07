NumPy ile minik bir autograd; amaç PyTorch'un in-place güvenliğini taklit etmek.

`Var(data)` hazır: `data` (ndarray), `version` (0), `grad`, `parents`, `_back` (geri fonksiyonu). Sen yaz:

- `mul(a, b)` — **girdileri** saklar: `(a, a.version)`, `(b, b.version)`.
- `exp(a)` — **çıktısını** saklar.
- `sum_(a)` — skaler.
- `add_(a, c)` — `a.data += c` (yerinde), `a.version += 1`. Grafı değiştirmez (sabit eklemek türevi değiştirmez).
- `backward(out)` — topolojik sıra, yapraklarda `grad` biriktir. Geri fonksiyonu sakladığı tensörün sürümü değişmişse `RuntimeError("... modified by an inplace operation")` fırlatsın.
