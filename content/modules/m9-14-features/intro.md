**Neden?** Görüntüler arasında nokta eşleştirmek (kayıt, panorama, 3-B) için ölçek, dönme, aydınlatma ve bakış açısı değişse de **aynı** noktaları bulan ve aynı biçimde **betimleyen** öznitelikler gerekir.

**Ölçek uzayı:** $L(x,y,\sigma)=G(x,y,\sigma)*f(x,y)$. SIFT ölçeği **oktavlara** böler (her oktav bir öncekinin yarı çözünürlüğü), her oktavı $s$ aralığa: $k=2^{1/s}$, oktav başına $s+3$ Gauss görüntüsü ($s+2$ DoG). Yeni oktavın ilk görüntüsü, öncekinin $2\sigma$'lık görüntüsünün ikiye alt örneklenmesidir.

**Anahtar noktalar:** $D(x,y,\sigma)=L(x,y,k\sigma)-L(x,y,\sigma)\approx(k-1)\sigma^2\nabla^2G*f$ (ölçek-normalize LoG). Her örnek, kendi ölçeğindeki 8 ve üstteki/alttaki ölçeklerdeki 9'ar komşusuyla (26) karşılaştırılır. Düşük kontrastlılar ($|D|<0.03$) ve kenar yanıtları ($\mathrm{Tr}^2(H)/\mathrm{Det}(H)\ge(r+1)^2/r$, $r=10$; Harris'teki fikir) atılır.

**Yön ve betimleyici:** anahtar noktanın ölçeğinde gradyanlardan 36 bölmeli yön histogramı, tepe = baskın yön. Betimleyici bu yöne döndürülmüş $16\times16$ örnekten: $4\times4$ alt bölge × 8 yön = **128** boyut; L2 normalize, 0.2'de kırp, tekrar normalize (doğrusal olmayan aydınlatmaya dayanıklılık).

**SURF:** Hessian $H=\begin{bmatrix}L_{xx}&L_{xy}\\L_{xy}&L_{yy}\end{bmatrix}$'nin determinantı; Gauss ikinci türevleri **kutu süzgeçlerle** ($9\times9$ ↔ $\sigma=1.2$) yaklaşıklanır, **integral görüntüyle** her boyutta sabit sürede: $\det\approx D_{xx}D_{yy}-(0.9D_{xy})^2$ (yanıtlar süzgeç alanına bölünür). Ölçek uzayı görüntüyü küçültmeden **süzgeci büyüterek** (9, 15, 21, 27, …) kurulur. Betimleyici: $20s$'lik pencerede $4\times4$ alt bölge, her birinde Haar yanıtlarından $(\sum d_x,\sum|d_x|,\sum d_y,\sum|d_y|)$ → 64 boyut.

**HOG:** merkezi fark gradyanları, $8\times8$ hücrede büyüklükle ağırlıklı 9 yönlü (0–180°) histogram, $2\times2$ hücrelik örtüşen bloklarda L2-Hys normalizasyonu, hepsini birleştir. 64 × 128 pencere: $7\times15$ blok × 4 hücre × 9 = 3780.
