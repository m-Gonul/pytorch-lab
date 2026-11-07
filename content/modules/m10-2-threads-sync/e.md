**Kendi thread havuzumuz** (C++): sabit işçi thread'leri, `std::queue<std::function<void()>>`, mutex + koşul değişkeni.

- `pool_sum(x, nthreads, chunk)` — `x` float64; her `chunk` elemanlık parça için bir görev, kısmi toplamlar diziye, ana thread bitişi bekler (atomik sayaç + cv). Toplam (double).
- `spawn_sum(x, nthreads)` — havuzsuz: nthreads adet thread aç, eşit böl.

Kontrol, görev ayrıntı düzeyini (chunk) tarar ve PyTorch'un kendi iç paralelliğiyle (`torch.set_num_threads`) karşılaştırır.
