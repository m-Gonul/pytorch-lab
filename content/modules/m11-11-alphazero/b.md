**PUCT araması**, saf Python, tic-tac-toe (tahta 9 elemanlı liste; 1 = X, −1 = O). "Ağ" bir fonksiyon: `net_fn(board, player)` → `(prior (9 liste), değer)` — değer **sıradaki oyuncunun** bakışından.

- `terminal(board)` → 1/−1/0/None.
- `puct_search(board, player, net_fn, n_sims, c=1.5)` → kökteki ziyaret sayıları (9 liste). Her düğüm: `P` (yasal hamlelere normalize önsel), `N`, `W` (kenar başına, düğümün oyuncusunun bakışından). Seçim $Q+c\,P\sqrt{\sum N+1}/(1+N)$ (ziyaretsiz kenarda Q = 0). Genişletmede çocuk tahtası terminal ise değer (sıradaki oyuncu için) −1 ya da 0; değilse `net_fn`. Geri yayılımda her seviyede işaret değişir.
- `visit_policy(N, tau)` — $N^{1/\tau}$ normalize; τ = 0 → argmax'a tek-sıcak (eşitlikte ilk).
