`slic(lab, K, c=10.0, iters=10)` — `lab (H, W, 3)` `float64` NumPy (L*a*b*); `(H, W)` `int64` etiket. Tanım (kontrol aynı tanımı döngülerle uygular, **birebir** eşleşmeli):

1. $S$ = `round(sqrt(H·W/K))`. Izgara: `y ∈ range(S//2, H, S)`, `x ∈ range(S//2, W, S)` (satır sırası).
2. Gradyan $G=\lVert I(y,x+1)-I(y,x-1)\rVert^2+\lVert I(y+1,x)-I(y-1,x)\rVert^2$ (`edge` dolgu); merkezi $3\times3$'te (görüntüye kırpılmış, satır sırası, ilk en küçük) en düşük $G$'ye taşı. Merkez $[L,a,b,y,x]$.
3. Her iterasyonda: piksel $p$ için adaylar $|y_p-c_y|\le S$ ve $|x_p-c_x|\le S$ olan merkezler; $D=\sqrt{d_c^2/c^2+d_s^2/S^2}$; etiket = en küçük $D$ (eşitlikte küçük indis); adayı olmayan piksel eski etiketini korur (başta −1).
4. Merkez = atanmış piksellerin $[L,a,b,y,x]$ ortalaması (boşsa eski).

Merkez hesabı 2. adımda döngülü olabilir; 3–4. adımlar tensör işlemleri (`index_add_`) ile, yalnız iterasyon döngüsü.
