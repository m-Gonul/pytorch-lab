**Dönüşüm çifti:** $T(u,v)=\sum_{x,y}f(x,y)\,r(x,y,u,v)$, $f(x,y)=\sum_{u,v}T(u,v)\,s(x,y,u,v)$. Çekirdek **ayrılabilir** ise $r=r_1(x,u)r_2(y,v)$, **simetrik** ise $r_1=r_2$: 2-B dönüşüm 1-B dönüşümlerle (önce satırlar, sonra sütunlar) hesaplanır. DFT çekirdeği ikisi de:

$$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)\,e^{-j2\pi(ux/M+vy/N)},\qquad f(x,y)=\frac1{MN}\sum_{u,v}F(u,v)\,e^{j2\pi(ux/M+vy/N)}.$$

**Neden frekans?** LTI/LSI sistemler konvolüsyonla tanımlanır ve karmaşık üsteller onların **özfonksiyonlarıdır**: $T[e^{j\omega t}]=H(\omega)e^{j\omega t}$. Bir sinyali üstellere ayırınca (Fourier = üstellere izdüşüm) sistemin etkisi frekans başına çarpmaya iner: **$f*h\Leftrightarrow F\cdot H$** ([[konvolüsyon teoremi]]). Bedeli: uzaysal konum bilgisi spektrumda görünmez, fazda saklıdır.

**Ayrık dünyanın tuzağı:** DFT periyodiktir, çarpım **dairesel** konvolüsyondur. Sarmalanmayı önlemek için iki diziyi $P\ge A+B-1$'e sıfırla doldur.

**Gonzalez hattı:** (1) $P=2M$, $Q=2N$ dolgu, (2) $(-1)^{x+y}$ ile merkezle, (3) DFT, (4) $G=H\cdot F$ (merkezli, simetrik $H$), (5) ters DFT, gerçek kısım, (6) $(-1)^{x+y}$, (7) sol üst $M\times N$'i kırp.

**Süzgeçler** ($D$: merkeze uzaklık): ideal $H=[D\le D_0]$ (halkalanır), Butterworth $\frac1{1+(D/D_0)^{2n}}$, Gauss $e^{-D^2/2D_0^2}$; yüksek geçiren $1-H_{LP}$.

**Restorasyon:** $G=HF+N$. Ters süzgeç $\hat F=G/H=F+N/H$: $H\to0$ olan yerde gürültü patlar. Wiener (en küçük ortalama kare): $\hat F=\frac{H^*}{|H|^2+S_\eta/S_f}G$, pratikte $S_\eta/S_f\approx K$ sabit.
