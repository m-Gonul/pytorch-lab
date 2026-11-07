**MCTS + vektörel rastgele oyunlar**, tic-tac-toe. Tahta 9 elemanlı liste (1 = X, −1 = O, 0 = boş).

- `winner(B)` — `B (N, 9)` tensör → `(N,)` 1/−1/0 (8 çizgi indeks tensörüyle, döngüsüz).
- `batched_playouts(board, player, n, g)` — sırası `player`'da olan tahtadan **n rastgele oyunu aynı anda** oyna (her hamlede `torch.multinomial` ile boş karelerden biri; biten oyunlar donar) → `(n,)` sonuç (X'in bakışından). Hamle döngüsü (≤ 9) serbest, oyun döngüsü yok.
- `mcts_move(board, player, n_sims, n_rollouts, g, c=1.4)` — UCT: seçim (sırası gelen oyuncunun bakışından $\bar Q\cdot\text{oyuncu}+c\sqrt{\ln N/n}$), genişletme (bir çocuk), simülasyon (`n_rollouts` paralel oyunun ortalaması ya da terminal değer), geri yayılım (X bakışından değer); en çok ziyaret edilen hamle.
