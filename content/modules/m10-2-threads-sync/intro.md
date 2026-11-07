**Süreç ve thread:** OS süreçleri zaman dilimleriyle döndürür (bağlam değişimi: zamanlayıcı kesmesi, G/Ç, sistem çağrısı). `fork()` yeni süreç yaratır: ayrı adres uzayı, iletişim için boru/paylaşımlı bellek — pahalı. **Thread** yalnız gerekeni (yığın, yazmaçlar) çoğaltır; kod, yığıt (heap) ve küresel değişkenler paylaşılır. Kullanım: performans (çekirdekler), arka plan işleri, asenkron işleme, daha iyi program yapısı.

**C++11 thread'leri** (pThreads modelini standartlaştırır): `std::thread t(f, args…)` hemen başlar; `t.join()` bekler, `detach()` bağımsız bırakır (ana thread bitince süreç biter). Argümanlar **kopyalanır**; referans için `std::ref`. Bir thread nesnesi join edilmeden yok edilirse program sonlanır (`std::terminate`).

**Paylaşılan veri:** salt okumalar güvenlidir. Bir thread okurken diğeri yazarsa? Banka hesabı: bakiye 10 000, maaş 1000 ve 500 aynı anda yatırılıyor; her iki thread "oku → topla → yaz" yaparsa sonuç 11 500, 11 000 ya da 10 500 olabilir.

- **Veri yarışı:** iki thread aynı konuma eşzamanlanmadan erişir ve en az biri yazar → C/C++'ta **tanımsız davranış** (8 bayttan büyük nesneler yırtılabilir, derleyici değeri yazmaçta tutabilir).
- **Yarış koşulu:** doğruluk zamanlamaya bağlı. Veri yarışı olmadan da olur (atomik oku + atomik yaz, arada başkası yazabilir).
- **Atomik:** `std::atomic<T>` tek işlemde oku–değiştir–yaz (`fetch_add`), sayaçlar ve biriktiriciler için ideal; yazma diğer thread'lerce görünür.
- **Karşılıklı dışlama:** `std::mutex` + `std::lock_guard`/`unique_lock` ile **kritik bölge**; çekişme yüksekse yavaş. Çoğu zaman en iyisi: **thread'e özel kopya** + sonda birleştirme.
- **Koşul değişkeni:** `cv.wait(lock, pred)` kilidi bırakıp uyur; `notify_one/all`. Sahte uyanmaya karşı yüklemle.
- **Kilitlenme (Coffman):** karşılıklı dışlama, tut-ve-bekle, zorla alınamama, döngüsel bekleme. Önlem: tüm kilitleri **aynı sırada** al (ya da `std::scoped_lock` ile birden).
- **Sahte paylaşım:** farklı thread'lerin farklı değişkenleri aynı 64 baytlık önbellek satırındaysa her yazma satırı diğer çekirdeklerde geçersizler → `alignas(64)`.
