Paralel programlama alıştırmalarının (BİL 455/533 Ex. 1–2) konusu: aynı çekirdeği tek thread'le ve `std::thread`'lerle yaz, thread sayısını (4, 8, en çok) ve problem boyunu (4096 tamsayı → milyonlar; 100×100 → büyük görüntüler) değiştirerek ölç.

**Integral dizi (önek toplamı):** $S_i=\sum_{j\le i}a_j$ — her eleman öncekine bağlı. **İki geçişli blok taraması:** (1) her thread kendi bloğunu yerel olarak tarar; (2) blok toplamları (p tane) taranır; (3) her thread bloğuna önceki blokların toplamını (ofset) ekler. İş ≈ 2n, yol ≈ 2n/p + p. (Genel paralel tarama M10.6.)

**Integral görüntü:** $I(y,x)=\sum_{y'\le y,x'\le x}f(y',x')$ = satır önek toplamları, sonra sütun önek toplamları. Satır geçişinde satırlar, sütun geçişinde sütun aralıkları bağımsız → iki paralel geçiş (arada bir bariyer: `join`). Uygulama: her boyutta O(1) kutu toplamı (Viola–Jones, M9.16).

**2B konvolüsyon** (alıştırma kuralları): girdi uint8, katsayılar fp32, hesap fp32, çıktı uint8 (yuvarla, [0, 255]'e kırp); filtre 3×3, 7×7, 11×11, 21×21. Geometrik ayrıştırma: çıktı satır bantları; her bant girdinin ±⌊k/2⌋ satırlık **halo**sunu da okur (salt okuma: yarış yok). Maliyet $HWk^2$; ayrılabilir filtrelerde $2HWk$.

**Matris çarpımı:** $2n^3$ FLOP, $3n^2$ veri → aritmetik yoğunluk büyük; ama naif ijk sırası B'nin sütununda yürür (önbellek dostu değil). **ikj** sırası ya da **devrik B** ardışık erişir; **karolama** önbellekte yeniden kullanımı artırır. Paralellik: C'nin satır blokları bağımsız.

Doğruluk kontrolü: "bir lineer cebir kütüphanesiyle karşılaştır" — burada PyTorch.
