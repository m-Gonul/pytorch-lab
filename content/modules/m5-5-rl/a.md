İki iterasyonda $V(s_1)$: $2\to3$; $V^*(s_1)=4$'e geometrik yaklaşıyor ($4-V_k=4\cdot0.5^k$). $s_0$'da en iyi eylem "git": $1+0.5\cdot4=3$, oysa "kal" $0+0.5\cdot3=1.5$.

Bandit'te $\nabla\log\pi(0)=e_0-\pi=(0.5,-0.5)$. Baseline tek bir örnekte tahmini değiştirir ($0.5\to0.2$), ama beklentiyi değiştirmez: eylem 1 örneklenseydi (ödül 0) $\theta_0$ bileşeni $(0-0.6)\cdot(-0.5)=0.3$ olurdu. Beklenti baseline'lı: $\tfrac12\cdot0.2+\tfrac12\cdot0.3=0.25$; baseline'sız: $\tfrac12\cdot0.5+\tfrac12\cdot0=0.25$. Aynı.
