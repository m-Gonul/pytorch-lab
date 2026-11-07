Saf Python, Sutton'ın 4 × 4 ızgarası: durumlar 0..15 (satır sırası), 0 ve 15 terminal (değer 0), eylemler yukarı/aşağı/sol/sağ (0..3), duvara çarpınca yerinde kal, her adım −1.

- `grid_step(s, a)` → `(s2, r)`.
- `evaluate(pi, gamma=1.0, theta=1e-10)` — `pi[s]` eylem olasılıkları listesi (4 eleman); **yerinde** (in-place, Gauss–Seidel) taramalarla, en büyük değişim `< theta` olunca dur. `(V, sweeps)`.
- `improve(V, gamma=1.0)` — açgözlü deterministik politika (olasılık listesi biçiminde, eşitlikte küçük eylem).
- `policy_iteration(gamma=1.0)` — rastgele politikadan başla; `(V, pi, n_improvements)`.
