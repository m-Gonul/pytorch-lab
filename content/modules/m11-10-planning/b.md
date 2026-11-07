**Dyna labirenti** (Sutton Şekil 8.2), saf Python. 6×9, duvarlar `{(1,2),(2,2),(3,2),(4,5),(0,7),(1,7),(2,7)}`, başlangıç (2,0), hedef (0,8) (ödül 1, bölüm biter). Eylemler yukarı, aşağı, sol, sağ; duvar/kenar → yerinde kal. γ = 0.95, α = 0.1, ε = 0.1, eşitlikler rastgele.

- `maze_step(s, a)` → `(s2, r, done)`.
- `dyna_q(n_eps, n_plan, seed=0)` → `(Q, bölüm başına adım listesi)`: her gerçek adımdan sonra Q-öğrenme, model[(s,a)] = (r, s′, done), sonra `n_plan` kez modelden rastgele **görülmüş** bir (s,a) seç ve simüle Q-öğrenme adımı.
