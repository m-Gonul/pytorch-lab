**Tarama:** girdi $x_0,\dots,x_{n-1}$, birleşmeli işleç ✦ ve birim I. **Kapsayıcı:** $y_i=x_0✦\dots✦x_i$; **dışlayıcı:** $y_0=I$, $y_i=x_0✦\dots✦x_{i-1}$ (kapsayıcı sağa bir kaydırılıp başa birim konmuş). Sıralı algoritma n − 1 işlem. Özyinelemeli gibi görünen pek çok hesap (her adım öncekine bağlı) tarama olarak yeniden yazılıp paralelleşir: kaynak ayırma, sıralama, ağaç işlemleri, polinom değerlendirme, doğrusal özyinelemeler.

**Naif paralel:** her çıktıyı ayrı bir thread toplar → $\sum_i i=n(n-1)/2$ işlem: daha çok iş, hızlanma yok.

**Kogge–Stone:** adım $s=1,2,4,\dots$: $y_i\leftarrow y_{i-s}✦y_i$ ($i\ge s$). $\log_2n$ adım; iş $\sum_k(n-2^k)=n\log_2n-(n-1)$. Yerinde güncelleme **yarış** üretir (thread i okumadan önce i − s yazmış olabilir: yazma-sonrası-okuma) → çift tampon (bir tampondan oku, ötekine yaz, rolleri değiştir) ya da oku → `__syncthreads` → yaz.

**Brent–Kung:** yukarı süpürme (indirgeme ağacı: $s=1,2,\dots$ için $y_{2s(k+1)-1}$'e $y_{\dots-s}$ eklenir) ve aşağı süpürme (kısmi toplamları ağaç boyunca geri dağıt). $2\log_2n-1$ adım, $2n-2-\log_2n$ işlem: sıralının sabit katı (**iş verimli**). Kaynak kısıtlıyken (n ≫ thread) BK, bolken KS kazanır.

**Hiyerarşik tarama:** n blok kapasitesini aşınca: (1) her blok kendi bölümünü tarar ve toplamını yazar, (2) blok toplamları taranır (gerekirse özyinelemeli), (3) her bölüme önceki toplam eklenir — M10.3'teki iki geçişli taramanın GPU hâli.

**Uygulamalar:** *akış sıkıştırma* (koşulu sağlayanları sıkıştır: bayrakların dışlayıcı taraması = çıkış konumu), *taban sıralaması* (her bitte 0'ları ve 1'leri tarama ile kararlı biçimde böl), *sayma sıralaması* (histogram + dışlayıcı tarama), *segmentli tarama*, *integral görüntü* (iki eksende tarama), *doğrusal özyineleme* (afin dönüşümlerin bileşkesi birleşmelidir).
