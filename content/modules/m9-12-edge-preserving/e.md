`nlm(img, h, tw=7, sw=21, sigma=0.0)` — `img (H, W)` float tensör. Hızlı NLM (ofset döngüsü):

1. `P = F.pad(img, t + s, mode="reflect")`, $t=tw/2$, $s=sw/2$.
2. Her arama ofseti $(d_y,d_x)\in[-s,s]^2$ için (tek döngü, ofset listesi üzerinde): kaydırılmış görüntüyle kare fark haritası; **kutu süzgeci** (`F.avg_pool2d(…, tw, stride=1)`) ile her pikselin yama MSE'si $d^2$ — yama boyutundan bağımsız maliyet.
3. $w=e^{-\max(d^2-2\sigma^2,0)/h^2}$; pay ve paydayı biriktir.

Karmaşıklık: naif $O(HW\cdot sw^2\cdot tw^2)$, bu yöntem $O(HW\cdot sw^2)$.
