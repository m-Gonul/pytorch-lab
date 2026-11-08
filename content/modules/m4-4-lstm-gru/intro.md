LSTM, RNN'in gizli durumuna ek olarak bir **hücre durumu** $c_t$ taşır ve onu kapılarla günceller. PyTorch'un formülü (kapı sırası **i, f, g, o**):

$$
\begin{aligned}
[a_i,a_f,a_g,a_o]&=W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh}\quad(4H\text{ boyutlu, tek matmul})\\
i&=\sigma(a_i),\quad f=\sigma(a_f),\quad g=\tanh(a_g),\quad o=\sigma(a_o)\\
c_t&=f\odot c_{t-1}+i\odot g,\qquad h_t=o\odot\tanh(c_t).
\end{aligned}
$$

Kritik nokta: $c_t$'den $c_{t-1}$'e giden yol **toplamsal** ve türevi yalnız $f$. Ağırlık matrisi ve doyurucu bir türev içermediği için gradyan uzun mesafe taşınır.

GRU iki kapıyla aynı fikri daha az parametreyle uygular (kapı sırası **r, z, n**):

$$
r=\sigma(W_{ir}x+b_{ir}+W_{hr}h+b_{hr}),\quad z=\sigma(\cdots),\quad n=\tanh\big(W_{in}x+b_{in}+r\odot(W_{hn}h+b_{hn})\big),\quad h'=(1-z)\odot n+z\odot h .
$$

:::türetim Hücre durumu boyunca gradyan neden sönmez?
$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$. Kapılar ve aday da $c_{t-1}$'e $h_{t-1}$ üzerinden bağlıdır, ama **doğrudan yol** yalnız birinci terimdir:
$$\frac{\partial c_t}{\partial c_{t-1}}=\mathrm{diag}(f_t)+(\text{}h_{t-1}\text{ üzerinden dolaylı terimler}).$$
Doğrudan yol boyunca $\frac{\partial c_T}{\partial c_k}\approx\prod_{t=k+1}^{T}\mathrm{diag}(f_t)$: bir ağırlık matrisiyle değil, **(0, 1) aralığındaki kapılarla** çarpım. Ağ $f_t\approx1$ öğrenirse gradyan neredeyse kayıpsız taşınır ("sabit hata karuseli", Hochreiter & Schmidhuber 1997); unutma kapısı bias'ını 1 ile başlatmak (Gers ve ark. 2000; Jozefowicz ve ark. 2015) bu yüzden önerilir. Düz RNN'de aynı çarpım $\mathrm{diag}(1-h^2)W$ idi (M4.3 P1) ve $W$'nin ölçeğine bağlı olarak sönüyordu.

**Sınırlar:** bu bir garanti değil, bir **olanaktır**: $f_t$ küçük öğrenilirse LSTM de unutur (bilerek). Dolaylı terimler patlayabilir; LSTM'de de gradyan kırpma kullanılır. GRU'da $h_t=(1-z_t)h_{t-1}+z_t\tilde h_t$, doğrudan yol çarpanı $1-z_t$ — aynı fikir, tek kapıyla.
:::
