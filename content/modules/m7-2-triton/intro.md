Triton'da bir **program**, verinin bir **blokunu** işler. Çekirdek, blok boyutunda vektörlerle yazılır:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

grid = (triton.cdiv(n, BLOCK),)
add_kernel[grid](x, y, out, n, BLOCK=1024)
```

- `tl.constexpr` derleme zamanı sabiti; `tl.arange` uzunluğu 2'nin kuvveti olmalı.
- **Maske**, blok tensörü aşınca dışarıyı okumayı/yazmayı engeller; `other=` maskelenen konumlara konacak değer.
- İşaretçi aritmetiği: `ptr + offs` bir işaretçi bloğu; 2-boyutlu bloklar için `ptr + rows[:, None] * stride + cols[None, :]`.

**GPU yoksa:** `TRITON_INTERPRET=1` ortam değişkeniyle (Triton import edilmeden **önce**) çekirdekler NumPy tabanlı bir yorumlayıcıda CPU tensörleri üzerinde çalışır. Hızlı değil ama sonuçlar aynı; doğruluğu burada, hızı GPU'da (Colab) sınarsın.
