NumPy ile Triton tarzı tiled matmul. Programlar `(pid_m, pid_n)`; her program `C`'nin bir `BM × BN` tile'ını hesaplar.

- `load_tile(M, r0, c0, R, Cc)` → `(R, Cc)` tile; matrisin dışına düşen kısımlar 0 (maskeli yükleme).
- `matmul_program(A, B, C, pid_m, pid_n, BM, BN, BK)` — K boyunca `BK` adımlarla `acc += load_tile(A, …) @ load_tile(B, …)`, sonra `C`'ye maskeli yaz.
- `matmul(A, B, BM=16, BN=16, BK=16)` — grid'i başlat. Program'lar üzerinde iki seviye döngü + programın içinde K döngüsü serbest; program içinde eleman döngüsü yok.
