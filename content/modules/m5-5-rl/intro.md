**MDP:** durumlar $S$, eylemler $A$, geçiş $P(s'\mid s,a)$, ödül $R(s,a)$, indirim $\gamma\in[0,1)$. Bir politikanın değeri $V^\pi(s)=\mathbb E_\pi\big[\sum_t\gamma^tr_t\mid s_0=s\big]$.

**Bellman denklemleri:**

$$
V^\pi=R_\pi+\gamma P_\pi V^\pi\ \Rightarrow\ V^\pi=(I-\gamma P_\pi)^{-1}R_\pi,\qquad
V^*(s)=\max_a\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\Big].
$$

**Value iteration:** ikinci denklemi sabit nokta iterasyonu olarak uygula.

Model bilinmiyorsa ortamla etkileşerek öğreniriz. **[[policy gradient]]** (REINFORCE):

$$
\nabla_\theta J=\mathbb E_{\tau\sim\pi_\theta}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t-b)\Big],\qquad G_t=\sum_{k\ge t}\gamma^{k-t}r_k .
$$

Kodda bu, `loss = -(log_prob * (G - b)).sum()` kaybının gradyanı.
