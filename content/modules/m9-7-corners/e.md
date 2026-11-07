`refine_subpix(img, q, win=5, iters=10)` — `img (H, W)` uint8 dizisi, `q (N, 2)` başlangıç köşeleri `[x, y]`; **tüm köşeler aynı anda**:

1. Her köşe için $(2\,win+3)^2$ noktalık ızgarayı `grid_sample` (bilineer, `align_corners=True`) ile örnekle.
2. İç $(2\,win+1)^2$ noktada merkezi fark / 2 ile $g=(g_x,g_y)$; ağırlık $w=e^{-(dx^2+dy^2)/(2(win/2)^2)}$.
3. $A=\sum w\,gg^T$, $b=\sum w\,gg^Tp$ ($p$ örnek noktasının koordinatı); $q\leftarrow A^{-1}b$ (`torch.linalg.solve` ile $N$ sistem birden).

Fikir: gerçek köşede, köşeden pencere noktasına giden vektör ya bir kenar üzerindedir ($g\approx0$ değilse ona dik) ya da düz bölgededir ($g=0$); her iki durumda $g^T(p-q)=0$. Yalnız `iters` döngüsü serbest.
