Saf Python, genel MDP: `P[a][s][s2]`, `R[s][a]`.

- `backup(P, R, V, gamma, s)` — tüm eylemler için $Q(s,a)=R(s,a)+\gamma\sum_{s'}P(s'|s,a)V(s')$ listesi.
- `finite_vi(P, R, VT, h, gamma=1.0)` → `(Vs, pis)`: `Vs[t]` $t=0..h$ ($V_h=VT$), `pis[t]` $t=0..h-1$ en iyi eylem indisleri (eşitlikte küçük indis).
- `retail_mdp()` → `(P, R, VT)`: notlardaki perakende problemi; eylem 0 yüksek fiyat, 1 indirim; $S_0$'da iki eylem de $S_1$'e götürür.
