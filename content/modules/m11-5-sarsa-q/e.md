1000 uçurum ortamı paralel (PyTorch). Q `(n_envs, 48, 4)`.

- `step(s, a)` — vektörel ortam: `(s2, r, done)` (uçurum → 36, −100).
- `batched_control(n_envs, n_steps, alpha, eps, method="q", seed=0)` — `method ∈ {"q", "sarsa", "expected"}`; her adımda her ortam bir adım atar; hedefe varan ortam başa döner (sonraki eylem yeniden seçilir). Çıktı `(Q, rew)`: `rew[t]` adım $t$'deki ortalama ödül.
- `greedy_returns(Q, max_steps=100)` → `(n_envs,)` açgözlü politikaların getirileri.

Adım döngüsü serbest; ortam döngüsü yok.
