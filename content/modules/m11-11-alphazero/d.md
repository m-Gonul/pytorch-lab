**Mini AlphaZero**, tic-tac-toe, PyTorch.

- `class Net` — 18 → 64 → 64 (ReLU), politika logit'leri (9) ve $\tanh$ değer. `encode(board, player)` → 18 (sıradaki oyuncunun taşları, rakibinkiler).
- `mcts(net, board, player, n_sims, rng, c=1.5, noise=True)` → kök ziyaret sayıları (NumPy). Kökte Dirichlet(0.3), ε = 0.25. Düğüm oluşturulurken ağ değerlendirilir (önsel yasal hamlelere maskelenir).
- `self_play(net, n_sims, rng)` → `[(board, player, π, z), …]`: ilk 3 hamlede π'den örnekle, sonra argmax.
- `train(net, opt, data, epochs=3, bs=64)` — 8 simetriyle artır, kayıp $(z-v)^2-\pi^\top\log p$ (L2 `weight_decay` ile).
- `alphazero(iters=15, games=20, n_sims=30, seed=0)` → ağ: her yinelemede `games` oyun, son 3000 örnek tamponu, Adam(2e-3, wd 1e-4).
