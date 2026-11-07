NumPy, `P (A, S, S)`, `R (S, A)`. Her algoritma `(V, pi, iters, backups)` döndürür; **backup** = bir durumun bir kez Bellman ile güncellenmesi (bir tarama = S backup; doğrudan çözüm ≈ S² backup sayılsın: muhasebe kuralı).

- `vi(P, R, gamma, eps)` — durma $\lVert V_n-V_{n-1}\rVert_\infty\le\varepsilon$.
- `pi_exact(P, R, gamma)` — doğrudan çözümlü PI; başlangıç politikası hep 0.
- `mpi(P, R, gamma, k, eps)` — iyileştir, $k$ değerlendirme taraması; durma VI'daki gibi (iyileştirme sonrası yeni değer ile önceki değer farkı).

Kontrol, hepsinin aynı $V^*$'a ulaştığını ve $k$'ya göre muhasebeyi tablolar.
