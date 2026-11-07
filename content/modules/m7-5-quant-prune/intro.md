**Simetrik kuantizasyon** ($b$ bit): $\ s=\max|x|/(2^{b-1}-1)$, $\ q=\operatorname{clamp}(\operatorname{round}(x/s))$, $\ \hat x=qs$.

**Afin** (zero-point'li, işaretsiz): $\ s=\frac{\max-\min}{2^b-1}$, $\ z=\operatorname{round}(-\min/s)$, $\ q=\operatorname{clamp}(\operatorname{round}(x/s)+z,\ 0,\ 2^b-1)$, $\ \hat x=(q-z)s$. Aralık sıfır etrafında simetrik değilse (ReLU sonrası aktivasyonlar) daha verimli.

**Ölçek tanecikliği:** tensör başına tek $s$; **kanal başına** her çıkış satırına bir $s$; **[[grup bazlı kuantizasyon]]** her satırı $g$ elemanlık gruplara bölüp her gruba bir $s$ (int4 LLM ağırlıklarında tipik $g=32\ldots128$).

**int8 matmul:** $\ y=x W^\top\approx(s_x s_w)\,(q_x q_W^\top)$; $q_xq_W^\top$ int32'de hesaplanır.

**Budama:** en küçük büyüklükteki ağırlıkları sıfırla. *Global* budama eşiği tüm katmanlar için tek; *katman başına* her katman aynı oranda.
