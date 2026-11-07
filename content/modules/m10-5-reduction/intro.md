**İndirgeme:** işleç ✦ (toplam, çarpım, min, max, …) ve birim eleman (0, 1, +∞, −∞) ile bir diziyi tek değere "özetle". Sıralı: `acc = birim; for x: acc = acc ✦ x` → N adım. **Ağaç:** 1. turda N/2 çift paralel, 2. turda N/4, … → $\log_2N$ adım; toplam iş $N/2+N/4+\dots+1=N-1$ (sıralıyla aynı: iş verimli). Yeterli kaynak varsa hızlanma N/log N.

**CUDA'da blok içi ağaç** (her thread iki eleman, paylaşımlı dizi):

- **Basit eşleme:** `i = 2*t; for (s = 1; s <= blockDim; s *= 2) { if (t % s == 0) in[i] += in[i + s]; __syncthreads(); }` — sahip konumlar 2s'nin katları; ikinci iterasyondan itibaren **her warp ayrışır**, son iterasyonlarda 31/32 kaynak boşa.
- **Yakınsak eşleme:** `i = t; for (s = blockDim; s >= 1; s /= 2) { if (t < s) in[i] += in[i + s]; __syncthreads(); }` — aktif thread'ler bitişik, warp'lar ya tamamen aktif ya pasif; ayrıca ardışık thread'ler ardışık adreslere erişir (birleştirilmiş, bank çakışmasız).
- **Paylaşımlı bellek:** ilk toplamayı global okumada yap, ara sonuçlar çip üstünde.
- **Hiyerarşi:** her blok kendi parçasını indirger, sonucu `atomicAdd` ile globale ekler (ya da ikinci çekirdek).
- **Kabalaştırma:** her thread önce C elemanı sıralı toplar → daha az blok, daha az senkronizasyon (iş aynı, yol biraz uzar — donanım zaten yetersizse bedava).
- **Warp karıştırma:** son 32 eleman paylaşımlı bellek olmadan `__shfl_xor_sync` kelebeğiyle.

**Histogram:** her girdi bir kutuya +1 — çıkış konumu veriye bağlı (dağıtım/scatter) → yarış. Global `atomicAdd`: aynı kutuya giden thread'ler sıralanır (tek renkli görüntüde felaket). **Özelleştirme:** blok başına paylaşımlı bellekte alt-histogram, sonunda globale birleştir; **birleştirme** (aggregation): ardışık aynı değerleri sayıp tek atomik.

**Kayan nokta birleşmeli değildir:** sıralı toplam O(Nε) hata (float32'de 2²⁴'ten sonra +1 kaybolur), ağaç (ikili) O(ε log N), Kahan telafili toplam O(ε). Paralel indirgemenin sonucu sıralıdan farklı — çoğu zaman daha doğru.
