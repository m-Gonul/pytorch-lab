PyTorch'ta aynı bilgiler: `t.shape`, `t.stride()`, `t.storage_offset()`, `t.is_contiguous()`, `t.untyped_storage().data_ptr()`.

- `layout(t)` → `(tuple(shape), stride(), storage_offset(), is_contiguous())`.
- `same_storage(a, b)` — iki tensör aynı storage'ı mı paylaşıyor?
- `flatten_transposed(x)` — `x.T`'yi düzleştir. `view` kullanırsan hata alırsın; doğru çağrıyı seç.
- `windows_torch(x, w)` — `x.unfold(0, w, 1)`; kopyasız `(n-w+1, w)`.
