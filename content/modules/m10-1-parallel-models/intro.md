**Neden paralellik:** Moore (1970'ler): yonga başına transistör sayısı ~2 yılda bir ikiye katlanır. Uzun süre frekans da arttı; ısı ve güç yüzünden ~2004–2005'te 2–4 GHz'de doydu. Artan transistörler artık çekirdeğe (IBM Power4 2001: ilk çift çekirdek) ve GPU'ya gidiyor. GPU çekirdeği tek başına zayıf ama sayıca çok: hem GFLOP/s'de hem **GFLOP/W**'de (enerji başına hesap) üstün. Bedeli: paralel kod — algoritmalar yeniden tasarlanmalı, yeni API'ler (CUDA, OpenCL).

**Flynn (1966):** SISD (sıralı), SIMD (bir komut çok veri: vektör birimleri, GPU'nun SIMT'si), MISD (pratikte yok), MIMD (en esnek). MIMD → **paylaşımlı bellek** (tek adres uzayı; ölçeklenince NUMA: yerel bellek hızlı, uzak bellek yavaş; simetrik ya da efendi–işçi — GPU'lu sistemler bu sonuncusuna benzer) ve **dağıtık bellek** (mesaj geçirme: kümeler, süperbilgisayarlar, veri merkezlerinde büyük model eğitimi). CPU: önbellek ve kontrol mantığı alanı kaplar; GPU: hesap birimleri. Ayrık GPU'da PCIe veri aktarımı bir **eşik boyut** yaratır; bu yüzden CPU+GPU'nun aynı belleği paylaştığı heterojen tasarımlar (Cell BE, AMD APU, Kepler SMX'leri, Xeon Phi).

**Hızlanma ölçüleri:** $S=T_1/T_p$, verimlilik $E=S/p$.

- **Amdahl:** seri oran $1-f$ sabit, problem sabit → $S=1/((1-f)+f/N)\le1/(1-f)$.
- **Gustafson:** p işlemciyle zaman sabit, problem büyür → $S=(1-f)+fN$.
- **Karp–Flatt:** ölçülen S'den deneysel seri oran $e=(1/S-1/p)/(1-1/p)$; p ile artıyorsa ek yük (iletişim, dengesizlik) var.
- **İş–yol:** iş W (tek işlemcide süre), yol $S_\infty$ (kritik yol), paralellik $W/S_\infty$. Açgözlü zamanlama (Brent): $\max(W/p,S_\infty)\le T_p\le W/p+S_\infty$.

**Tasarım (Foster'ın PCAM'i):** **Bölümleme** — işlevsel (görev) ya da veri (alan) ayrıştırması; işlemci sayısının 1–2 mertebe fazlası kadar görev. **İletişim** — görev grafiği; veri bağımlılıkları. **Birleştirme** — iletişimi yok etmek için görevleri grupla (yine işlemciden fazla grup). **Eşleme** — yükü dengele, pahalı iletişimi aynı düğüme koy (NP-tam; sezgisel). Hesap tekrarlamak iletişimden ucuz olabilir.

**Ayrıştırma desenleri** (Mattson): görev paralelliği, böl ve yönet (merge sort), geometrik ayrıştırma (dizi/ızgara blokları + kenar "halo"), özyinelemeli veri (kısmi toplamlar: giderek büyüyen aralıklar), boru hattı (aşama süreleri dengeli olmalı), olay tabanlı koordinasyon (düzensiz iletişim, ayrık olay simülasyonu). **Program yapısı desenleri:** küresel paralel–yerel sıralı (SPMD, MPMD, efendi–işçi, map-reduce) ve küresel sıralı–yerel paralel (fork–join, döngü paralelliği).
