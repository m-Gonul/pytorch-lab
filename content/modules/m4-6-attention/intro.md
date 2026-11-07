Her token için üç vektör: **query** $q$ (ne arıyorum), **key** $k$ (neyim), **value** $v$ (ne veririm). Scaled dot-product attention:

$$
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}+M\Big)V,
$$

$M_{ij}=-\infty$ ise $i$, $j$'yi görmez ([[causal maske]]: $j>i$). Çıktının $i$. satırı, $V$ satırlarının $A_{i,:}$ ağırlıklarıyla ortalaması.

**Multi-head:** $d$ boyutunu $h$ parçaya böl ($d_k=d/h$), her head kendi attention'ını yapsın, sonuçlar birleştirilip $W_O$ ile karıştırılsın. Kodda bu, `(B, T, d)` → `(B, h, T, d_k)` reshape'i ve tek bir batch'li matmul.

**Online softmax:** $\operatorname{softmax}$'ı bloklar hâlinde, satır başına koşan bir maksimum $m$ ve payda $\ell$ tutarak hesaplayabilirsin. Yeni blokla $m'=\max(m,\max_j s_j)$:

$$
\ell'=\ell\,e^{m-m'}+\sum_j e^{s_j-m'},\qquad o'=o\,e^{m-m'}+\sum_j e^{s_j-m'}v_j,\qquad \text{sonuç}=o/\ell.
$$
