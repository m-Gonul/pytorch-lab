**Kenar modelleri:** *basamak* (bir pikselde geçiş), *rampa* (lens ya da sıkıştırma bulanıklığıyla yayılmış basamak), *çatı* (ince çizgi). Bir rampada birinci türev rampa boyunca pozitif, düz bölgede sıfırdır; ikinci türev rampanın başında pozitif, sonunda negatiftir; ikisinin arasındaki **sıfır geçişi** kalın kenarın merkezini verir.

**Gürültü** türevlerle büyür: $\sigma^2=0.1$ kadar küçük bir Gauss gürültüsü ikinci türevi tanınmaz hale getirir. Bu yüzden her dedektör üç adımdır: **yumuşat → aday noktaları bul → yerelleştir.**

**Gradyan:** $\nabla f=(g_x,g_y)$, büyüklük $M=\sqrt{g_x^2+g_y^2}$ (ya da $|g_x|+|g_y|$), yön $\alpha=\operatorname{atan2}(g_y,g_x)$. Kenarın kendisi gradyana diktir. Operatörler: Roberts (2 × 2), Prewitt, Sobel (merkezde 2 ağırlığıyla ek yumuşatma), **Kirsch** (8 pusula yönü, en güçlü yanıt seçilir).

**Marr-Hildreth:** $g=\nabla^2G*f$, ardından sıfır geçişleri. [[LoG]] $\nabla^2G=\frac{x^2+y^2-2\sigma^2}{\sigma^4}e^{-\frac{x^2+y^2}{2\sigma^2}}$ ("Meksika şapkası"), sıfırları $r=\sqrt2\sigma$ çemberinde. Çekirdek boyutu $n$: $\lceil6\sigma\rceil$'dan büyük ya da eşit en küçük tek sayı. Sıfır geçişi: $3\times3$ komşulukta dört karşıt çiftten (sol/sağ, üst/alt, iki çapraz) en az birinin işaretleri farklı **ve** farkları eşikten büyük.

**Canny** (1986; düşük hata, iyi yerelleşme, tek yanıt): 1-B gürültülü basamak için optimal dedektörün iyi bir yaklaşığı **Gauss'un türevidir**. 2-B'de:

1. Gauss ile yumuşat.
2. Gradyan büyüklüğü ve açısı.
3. [[non-maxima suppression]]: açıyı 4 yöne nicemle (0°, 45°, 90°, 135°; ±22.5° aralıklar), gradyan yönündeki iki komşudan küçükse sıfırla.
4. [[histerezis eşikleme]]: $T_H$ üstü güçlü, $[T_L,T_H)$ zayıf; güçlü bir piksele 8-komşulukla bağlı zayıflar kalır. Canny $T_H:T_L$ = 2:1 ile 3:1 önerir.
