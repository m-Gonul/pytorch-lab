**Neden yaklaştırma:** Go ~$3^{361}$ durum, Atari durumu 210×160×3 ekran; sürekli durumlar ayrıklaştırılınca patlar. Durumu öznitelik vektörüyle temsil et, $s=(x_1,\dots,x_N)$, ve öğren: doğrusal $Q(s,a)\approx\sum_iw_{a,i}x_i$ ya da sinir ağı $Q(s,a)\approx q(x;w)$.

**Kayıp:** $\tfrac12\big[Q_w(s,a)-(r+\gamma\max_{a'}Q_{w}(s',a'))\big]^2$. Sorun: **w hem kestirimde hem hedefte**; w'yi değiştirince hedef de kayar. Yarı-gradyan yöntemler hedefi sabit sayar (gradyanını almaz); DQN bunu sertleştirir: hedefte ayrı, yavaş güncellenen ağırlıklar $\bar w$.

**Yakınsama:** tablo Q-öğrenme ve doğrusal Q-öğrenme (politika üstü dağılımda) Robbins–Monro koşullarıyla yakınsar. Doğrusal olmayan ağlarla ve politika dışı veriyle garanti yok — **ölümcül üçlü** (fonksiyon yaklaştırma + önyükleme + politika dışı eğitim) ıraksayabilir (Baird 1995).

**DQN = gradyan Q-öğrenme + derin ağ + iki hile:**

1. **Deneyim tekrarı:** $(s,a,r,s')$'leri bir tamponda sakla, her adımda rastgele mini-batch ile öğren. Ardışık örneklerin korelasyonunu kırar, her deneyimi birçok kez kullanır (veri verimliliği), davranış dağılımını eski politikalar üzerinde ortalar (geri besleme döngüsünü önler). Tampondaki veri eski politikalardan geldiği için öğrenme **politika dışı** olmalı.
2. **Hedef ağ:** $\bar w$ her C adımda bir $w$'ye kopyalanır (ya da yumuşak: $\bar w\leftarrow\tau w+(1-\tau)\bar w$).

Ek: Huber kaybı (büyük hatalarda gradyan kırpma), ödül kırpma ($\mathrm{sign}(r)$), kare atlama, ε 1→0.1 doğrusal azalma. **Çift DQN:** seçimi çevrimiçi, değerlendirmeyi hedef ağla yap.
