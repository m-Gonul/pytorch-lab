Saf Python. Ortam gym tarzı: `P[s][a]` = `(olasılık, sonraki_durum, ödül, bitti)` listesi; `bitti` ise sonraki durumun değeri 0 sayılır.

- `q_values(P, V, s, nA, gamma)` → `[Q(s, a) for a]`.
- `value_iteration(P, nS, nA, gamma, tol=1e-10)` → `(V, policy)`; $\max_s|V_{k+1}-V_k|<$ `tol` olunca dur. `policy[s]` = argmax (eşitlikte küçük indeks).
- `policy_evaluation(P, policy, nS, gamma, tol=1e-10)` → deterministik `policy` için $V^\pi$ (iteratif).

Kontrol 4×4 kaygan bir buzlu göl (FrozenLake benzeri) üzerinde çalışır.
