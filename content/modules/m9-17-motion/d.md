Yoğun, piramitli Lucas–Kanade (PyTorch).

- `warp(img, flow)` — `img (H, W)`, `flow (2, H, W)`: $img(x+u, y+v)$, `grid_sample` (`align_corners=True`, `padding_mode="border"`).
- `lk_level(I0, I1, flow, win=7, iters=5)` — $I_0$'ın gradyanları (merkezi fark / 2, `replicate`), pencere ortalamaları (`avg_pool2d`, `replicate` dolgu); her iterasyonda $I_t=$`warp(I1, flow)`$-I_0$, $\lambda_{min}>10^{-2}$ olan yerlerde LK adımı (adım ±1 pikselle kırpılır), ardından akışı **güvenle ağırlıklı** pencere ortalamasıyla düzgünleştir: $f\leftarrow\frac{\text{box}(\lambda_{min}f)}{\text{box}(\lambda_{min})}$ (yoksa düz bölgelerin anlamsız akışları komşu pencerelerin $I_t$'sini bozar).
- `pyr_lk(I0, I1, levels=4, win=7, iters=5)` — `avg_pool2d` ile piramit; kabadan inceye, seviye değişirken `F.interpolate(…, mode="bilinear")` ve ×2.
- `fb_error(f01, f10)` — $\lVert f_{01}(x)+f_{10}(x+f_{01}(x))\rVert$ (`warp` ile).
- `fruc_mc(I0, I2, flow02)` — ara kare $\frac12[I_0(x-\frac12f)+I_2(x+\frac12f)]$ (`warp` ile; akışın ara noktada da aynı olduğu yaklaşımı).
