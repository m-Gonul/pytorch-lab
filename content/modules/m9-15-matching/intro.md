**Şablon eşleme:** $w$ şablonunu $f$ üzerinde kaydır. Ham korelasyon $\sum w\,f$ parlak bölgelerde her zaman büyüktür; bunun yerine **korelasyon katsayısı**

$$\gamma(x,y)=\frac{\sum_{s,t}[w(s,t)-\bar w]\,[f(x+s,y+t)-\bar f_{xy}]}{\sqrt{\sum[w-\bar w]^2\sum[f-\bar f_{xy}]^2}}\in[-1,1]$$

genlik (kazanç) ve ofset değişimine değişmezdir. En büyük $\gamma$ en iyi eşleşmedir. Dönme ve ölçek için şablonun dönmüş/ölçeklenmiş kopyaları gerekir; ya da SIFT gibi değişmez öznitelikler (M9.14) eşleştirilir.

**Homografi:** düzlemsel sahne ya da yalnız dönen kamera için iki görüntü arasındaki dönüşüm $\tilde x'=H\tilde x$ ($3\times3$, ölçek serbest → 8 serbestlik). Her nokta eşi iki doğrusal denklem verir: **DLT** $A\mathbf h=0$, $\mathbf h$ = $A$'nın en küçük tekil değerine ait sağ tekil vektör; 4 nokta yeter, fazlası en küçük kareler. **Hartley normalizasyonu:** noktaları ağırlık merkezi 0, ortalama uzaklık $\sqrt2$ olacak şekilde ölçekle; aksi halde $A$'nın sütunları $10^6$ kat farklı ölçeklerde olur, çözüm kararsızlaşır.

**[[RANSAC]]:** eşleşmelerin önemli bir kısmı yanlıştır. Tekrarla: rastgele 4 eş, $H$, eşik içinde kalan **içerdeğerleri** say; en çok içerdeğerli hipotezi seç ve yalnız onlarla yeniden kestir. Gereken iterasyon $N=\lceil\log(1-p)/\log(1-w^s)\rceil$.

**Korelasyon süzgeçleri (DCF):** şablonu sabit tutmak yerine, hedef üzerinde istenen yanıtı (merkezde Gauss tepe) veren süzgeci **öğren**. Frekans uzayında kapalı çözüm (MOSSE): $H^*=\frac{\sum_iG_i\odot F_i^*}{\sum_iF_i\odot F_i^*+\lambda}$; her karede $A_t=\eta GF^*+(1-\eta)A_{t-1}$, $B_t=\eta FF^*+(1-\eta)B_{t-1}$ ile güncelle. Ön işleme: log, normalize, kosinüs (Hann) penceresi. Güven: tepe/yan lob oranı **PSR** $=\frac{g_{max}-\mu_{yan}}{\sigma_{yan}}$. Sınırlar: dairesel korelasyon gerçek dışı kaydırılmış örneklerle eğitir, dikdörtgen olmayan hedefte arka planı öğrenir — CSR-DCF (CSRT) uzamsal ve kanal güvenilirliğiyle bunları çözer.
