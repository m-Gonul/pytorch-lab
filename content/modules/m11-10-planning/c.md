**Kesinlik eşdeğerli planlama**, NumPy. Rastgele bir MDP'den (`P (A,S,S)`, `R (S,A)` belirli) düzgün rastgele politikayla toplanan geçişler `(S_, A_, R_, S2_)`.

- `fit_model(S_, A_, R_, S2_, nS, nA)` → `(P_hat, R_hat)`: sayımlar (`np.add.at`); görülmemiş (s,a) için kendine geçiş ve ödül 0.
- `plan(P, R, gamma, n_iter=500)` → Q (değer yinelemesi, yalnız yineleme döngüsü).
- `q_learning_replay(S_, A_, R_, S2_, nS, nA, gamma, epochs=1, alpha=0.1)` — aynı veri üzerinde tablo Q-öğrenme (epoch kadar tekrar; döngü serbest).
- `policy_value(P, R, pi, gamma)` → V (doğrusal sistem).
