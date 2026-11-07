**Bağlamsal haydut:** durumlar $S$, her $s$ bir öznitelik vektörü $x_s$ (konum, yaş…); eylemler $A$, her $a$ bir $x_a$ (konu, anahtar kelimeler…). Geçiş yok — durumlar bağımsız gelir. Amaç: $\pi:x_s\to a$, $\mathbb E[r|x_s,x_a]$'yı büyüt. Uygulama: kişiselleştirilmiş reklam ve web içeriği (MSN haberleri bağlamsal haydutlarla tıklama oranında %26 artış bildirdi, 2017).

**Ödülü regresyonla öğren:** $\tilde R(s,a)=\tilde R(x)$, $x=(x_s,x_a)$; doğrusal $w^\top x$ ya da sinir ağı.

**Bayesçi doğrusal regresyon:** önsel $N(w|0,\lambda^2I)$, olabilirlik $N(r|w^\top x,\sigma^2)$ → sonsal Gauss:

$$\Sigma=\Big(\sigma^{-2}\sum_ix_ix_i^\top+\lambda^{-2}I\Big)^{-1},\qquad \mu=\sigma^{-2}\,\Sigma\sum_ix_ir_i.$$

Yeni $x$ için öngörü: $N\big(r\,\big|\,x^\top\mu,\;x^\top\Sigma x+\sigma^2\big)$ (ilk terim parametre belirsizliği, ikincisi gürültü).

- **UCB:** $x^\top\mu+c\sqrt{x^\top\Sigma x}$, $c=1+\sqrt{\ln(2/\delta)/2}$ (LinUCB).
- **Thompson:** $\tilde w\sim N(\mu,\Sigma)$, $a=\arg\max_a\tilde w^\top x_a$ (ya da $k$ örnek ortalaması).

Her adımda yalnız seçilen eylemin verisi gelir; $\Sigma^{-1}$'e $xx^\top/\sigma^2$ eklenir. **Sherman–Morrison:** $(A+uv^\top)^{-1}=A^{-1}-\dfrac{A^{-1}uv^\top A^{-1}}{1+v^\top A^{-1}u}$ — ters matrisi $O(d^2)$'de günceller.

**Kayıtlı veriden değerlendirme:** yeni bir politikayı canlıya almadan, eski politikanın kayıtlarıyla ($x$, $a$, $r$, $p(a|x)$) ölç: DM (ödül modeli), IPS ($\mathbb 1[\pi(x)=a]\,r/p$), SNIPS, DR; ya da tekrar oynatma (yalnız eylemler eşleşince say).
