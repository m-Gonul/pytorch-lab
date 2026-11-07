**Uçurum yürüyüşü** (Sutton Örnek 6.6), saf Python. 4×12 ızgara, durum `s = satır·12 + sütun`; başlangıç 36 (sol alt), hedef 47 (sağ alt); 37..46 uçurum. Eylemler 0 yukarı, 1 sağ, 2 aşağı, 3 sol; ızgara dışına çıkılmaz. Her adım −1; uçuruma basmak −100 ve başlangıca dönüş (bölüm sürer).

- `cliff_step(s, a)` → `(s2, r, done)`.
- `eps_greedy(Q, s, eps, rng)` — eşitlikleri rastgele boz.
- `sarsa(n_eps, alpha=0.5, eps=0.1, seed=0)` ve `qlearning(...)` → `(Q, getiriler)`; Q `48 × 4` liste.
- `greedy_return(Q, max_steps=100)` — açgözlü politikanın getirisi.
