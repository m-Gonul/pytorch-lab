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
