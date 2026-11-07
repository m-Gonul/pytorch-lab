Büyük MDP'de politika değerlendirmesi.

- `random_sparse_mdp(S, A, k, seed=0)` — her (s, a) için rastgele k hedef durumlu seyrek geçiş; `torch.sparse_coo` ya da (A, S, k) indeks + olasılık dizileri.
- `eval_direct(Pi_dense, r, gamma)` — `torch.linalg.solve`.
- `eval_iterative(idx, prob, r, gamma, tol)` — $V\leftarrow r+\gamma\sum_kp_k V[\text{idx}_k]$ (gather), yakınsayana dek; tarama sayısını da döndür.
- `mpi_sparse(idx, prob, R, gamma, k, eps)` — seyrek MPI.

Kontrol, S büyüdükçe doğrudan çözümün ($O(S^3)$) ve seyrek yinelemenin ($O(Sk)$ / tarama) süresini karşılaştırır.
