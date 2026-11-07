Triton (GPU yoksa yorumlayıcı).

- `conv2d(img, k, BH=16, BW=16)` — `img` uint8 `(H, W)`, `k` fp32 `(K, K)`; her program BH×BW çıktı karosu; K² dokunuş için kaydırılmış girdi bloğunu (maskeli, dışı 0) yükle ve biriktir; çıktı $\min(255,\max(0,\lfloor acc+0.5\rfloor))$ uint8.
- `matmul(a, b, BM=32, BN=32, BK=32, GROUP_M=4)` — fp32 (int32 girdi fp32'ye çevrilir); program kimliğini **gruplanmış sırada** (GROUP_M satır bloğu × tüm sütunlar) eşle; `tl.dot`; kenar maskeleri.
