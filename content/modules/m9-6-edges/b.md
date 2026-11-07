Saf Python, liste listeleri; görüntü dışında kalan komşular **0** sayılır.

- `nms(mag, gx, gy)` — $\alpha$ = `degrees(atan2(gy, gx)) % 180`; $[0,22.5)\cup[157.5,180)$ → sol/sağ; $[22.5,67.5)$ → $(y-1,x-1)$ ve $(y+1,x+1)$ (y aşağı!); $[67.5,112.5)$ → üst/alt; $[112.5,157.5)$ → $(y-1,x+1)$ ve $(y+1,x-1)$. İki komşudan **herhangi birinden küçükse** 0, değilse büyüklük.
- `hysteresis(gn, lo, hi)` — güçlü $\ge$ `hi`, zayıf $[$`lo`, `hi`$)$; güçlülerden 8-komşulukla genişleyen BFS (`collections.deque`). Çıktı 0/1.
- `zero_crossings(g, thr)` — kenarda olmayan her piksel için dört karşıt çift; $g_ag_b<0$ ve $|g_a-g_b|>$ `thr` ise 1. Görüntü kenarı 0.
