Triton (GPU yoksa yorumlayıcı). Dosyanın başı hazır: GPU yoksa `TRITON_INTERPRET=1`, `DEVICE` seçimi.

- `add_kernel` + `add(x, y)` — `BLOCK = 256`.
- `softmax_kernel(X, Y, stride, n_cols, BLOCK)` + `softmax(x)` — program başına bir satır, `BLOCK = triton.next_power_of_2(n_cols)`; satırı yükle (maskeli, `other=-inf`), max çıkar, exp, toplam, böl, sakla.

Triton paket olarak yalnız Linux x86-64'te var; macOS/Windows'ta bu aşamayı Colab'da çalıştır.
