<!-- çözüm -->

Önce olabilirliği iki farklı $p$ için doğrudan çarpımla hesapla: $p=0.75$ daha yüksek çıkmalı, çünkü veride 4 atışın 3'ü tura. Sonra $\log L$'nin türevini sıfıra eşitle:

$$
\ell(p)=k\ln p+(n-k)\ln(1-p),\qquad \ell'(p)=\frac kp-\frac{n-k}{1-p}=0\;\Rightarrow\;\hat p=\frac kn .
$$

Gauss'ta ortalama ve iki varyansı karşılaştır. Zarda 2 hiç gelmediği için MLE ona **sıfır** olasılık veriyor: model, hiç görmediği bir sonucu imkânsız sayıyor. Bu, küçük veride MLE'nin aşırı uyumunun en sade örneği.

Tablo: $\ln 0.75\approx-0.28768$, $\ln 0.25\approx-1.38629$.
