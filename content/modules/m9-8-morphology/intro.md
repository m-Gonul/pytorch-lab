**Kümeler ve yapı elemanları (SE):** ikili görüntüde nesne $A$ = ön plan pikselleri. SE küçük bir kümedir, bir **orijini** vardır (genelde merkez). Yansıma $\hat B=\{-b\mid b\in B\}$ (180° döndürme), öteleme $(B)_z=\{b+z\}$.

**Erozyon:** $A\ominus B=\{z\mid(B)_z\subseteq A\}$ — SE'nin tamamen sığdığı yerler; nesneler küçülür, $B$'den ince bağlar kopar.

**Genişleme:** $A\oplus B=\{z\mid(\hat B)_z\cap A\ne\varnothing\}=\{a+b\}$ (Minkowski toplamı); nesneler büyür, boşluklar dolar.

**İkilik:** $(A\ominus B)^c=A^c\oplus\hat B$, $(A\oplus B)^c=A^c\ominus\hat B$.

**Açma** $A\circ B=(A\ominus B)\oplus B=\bigcup\{(B)_z\mid(B)_z\subseteq A\}$: konturu yumuşatır, ince çıkıntıları ve köprüleri siler. **Kapama** $A\bullet B=(A\oplus B)\ominus B$: dar boşlukları ve küçük delikleri kapatır. Özellikler: $A\circ B\subseteq A\subseteq A\bullet B$; ikisi de **idempotent** ($(A\circ B)\circ B=A\circ B$).

**[[hit-or-miss]]:** $I\circledast B_{1,2}=\{z\mid(B_1)_z\subseteq A\ \text{ve}\ (B_2)_z\subseteq A^c\}$; tek SE'de ön plan (1), arka plan (0) ve **umursamaz** elemanlarla da yazılır. Algoritmalar: sınır $A-(A\ominus B)$, delik doldurma, bağlı bileşenler, dışbükey zarf, inceltme, iskelet, budama.

**Gri seviye**, düz SE: erozyon $=\min_{(s,t)\in b}f(x+s,y+t)$, genişleme $=\max_{(s,t)\in b}f(x-s,y-t)$. Düz olmayan: $[f\ominus b_N]=\min(f(x+s,y+t)-b_N(s,t))$, $[f\oplus b_N]=\max(f(x-s,y-t)+b_N(s,t))$. **Gradyan** $(f\oplus b)-(f\ominus b)$; **top-hat** $f-(f\circ b)$; **bottom-hat** $(f\bullet b)-f$. Geometrik yorum: açma, SE'yi yüzeyin **altından** yukarı itip ulaştığı en yüksek noktaları alır.
