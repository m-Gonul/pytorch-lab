C++ (`std::thread`) — CPU'da indirgeme ve histogramın ince noktaları.

- `sum_acc(x, nthreads, k)` — float32 dizi, thread başına **k ayrı biriktirici** (döngüyü k'lık açıp `acc[j] += x[i + j]`), sonunda birleştir; double döndür. k = 1'de her toplama bir öncekini bekler (gecikme zinciri), k ≥ 4 komut düzeyi paralellik (ILP) sağlar.
- `hist_copies(img, nthreads, copies)` — thread başına `copies` adet alt-histogram (piksel i → kopya i mod copies); tek renkli görüntüde aynı kutuya art arda yazma zinciri kırılır.
