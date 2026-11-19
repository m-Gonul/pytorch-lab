**Üç halka denklemini yazın.** A→B ve B→C ödül0, C→A ödül1. Bir tur üç adım: ρ=1/3. Diferansiyel Bellman denklemi her durumda `h=reward−rho+h_next`; çözüm sabite kadar belirlenir. `µᵀh=0` seçince h=(−1/3,0,+1/3).

**Limitleri ayırın.** +1,0 dönüşümlü ödülde ortalama1/2 ama merkezli toplam salınır. Abel hesabında önce sabit γ<1 ile sonsuz geometrik toplamı, sonra γ↑1 limitini alın.

**İzde n katsayısını kaçırmayın.** n2, başlangıç gain10, toplam ödül2 ve Q farkı0 için δ=−18. İlk iz o=β; ham δ kullanan naif güncelleme−8, ortalama hedefini kullanan doğru güncelleme1 verir. Aşağıdaki alanları bu denklemlerle doldurun.
