GPT, token + konum embedding'inin üstüne $L$ tane aynı yapıda blok koyar ve en sonda sözlük üzerinde logit üretir:

$$
\begin{aligned}
x_0 &= W_{te}[\text{idx}] + W_{pe}[0..T),\\
x &\leftarrow x + \operatorname{MHA}_{\text{causal}}(\operatorname{LN}_1(x)),\\
x &\leftarrow x + \operatorname{MLP}(\operatorname{LN}_2(x)),\quad \operatorname{MLP}(u)=W_2\,\operatorname{GELU}(W_1u),\\
\text{logits} &= \operatorname{LN}_f(x)\,W_{te}^\top.
\end{aligned}
$$

Son satırda çıkış katmanı ayrı bir matris değil, $W_{te}$'nin kendisi ([[weight tying]]): GPT-2 small'da 38.6M parametre tasarrufu.

Blok başına parametre: qkv $3d^2+3d$, proj $d^2+d$, MLP $8d^2+5d$, iki LN $4d$; toplam $12d^2+13d$.

**Örnekleme:** logit $z$, temperature $\tau$, top-k: en büyük $k$ logit dışındakileri $-\infty$ yap, sonra $p=\operatorname{softmax}(z/\tau)$.

**[[kv-cache]]:** üretimde her adımda yalnız son token'ın query'si gerekir; geçmiş token'ların key/value'larını saklarsan her adım tek bir token'lık iş olur.
