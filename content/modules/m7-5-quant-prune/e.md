LLM tarzı ağırlık-yalnız **int4**, [[grup bazlı kuantizasyon]], **döngü yok**:

- `quantize_int4(W, group)` → `(q, s)`: `W (out, in)`, `in` `group`'a bölünür; `W.view(out, in/group, group)`, grup başına $s=\max|w|/7$, $q\in[-8,7]$ (int8 tensörde).
- `pack_int4(q)` → `uint8 (out, in/2)`: iki değeri bir bayta: `(q[..., 0::2] & 0xF) | ((q[..., 1::2] & 0xF) << 4)`.
- `unpack_int4(p)` → `int8`; 4 bitlik iki tümleyeni geri işaretle (`v - 16 if v > 7`).
- `dequant_int4(p, s, group)` → fp32 `W`.

Kontrol yeniden yapılandırma hatasını grup boyutuna göre ve bit/ağırlık maliyetini (ölçekler dahil) tablolar.
