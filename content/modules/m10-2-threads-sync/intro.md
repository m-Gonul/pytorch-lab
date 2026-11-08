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

:::ayrıntı Bellek tutarlılığı: sıralı tutarlılık ve doğrusallaştırılabilirlik
**Sıralı tutarlılık** (Lamport 1979): bir çok thread'li çalıştırmanın sonucu, tüm thread'lerin işlemlerinin *bir* serpiştirilmesiyle (her thread kendi program sırasında) elde edilebilen sonuçlardan biridir. B aşamasındaki doğrulayıcımız tam olarak bu modeli varsayar. Gerçek donanım daha zayıftır: x86 TSO'da bir thread'in yazması kendi depolama tamponunda bekleyebilir, bu yüzden Peterson algoritması (E2) bellek bariyeri olmadan bozulur. C++'ta varsayılan `std::atomic` işlemleri (`memory_order_seq_cst`) sıralı tutarlılık sağlar; `relaxed`/`acquire`/`release` daha ucuz ama daha zayıftır.

**Doğrusallaştırılabilirlik** (Herlihy & Wing 1990): eşzamanlı bir nesnenin (kuyruk, yığın, sayaç) her işlemi, çağrısı ile dönüşü arasında bir anda "atomik olarak gerçekleşmiş" gibi görünür ve bu anlar gerçek zaman sırasına uyar. Sıralı tutarlılıktan güçlüdür (gerçek zaman sırasını da korur) ve **birleştirilebilir**dir: her nesne ayrı ayrı doğrusallaştırılabilirse bütün sistem de öyledir. E5'teki kilitsiz yığının doğrusallaştırma noktası başarılı CAS anıdır.
:::

:::ayrıntı Kullanıcı düzeyi thread'ler ve fiber'ler
İşletim sistemi thread'i çekirdek tarafından zamanlanır: her bağlam değişimi bir sistem çağrısı ve önbellek kaybı demektir (mikrosaniyeler). **Kullanıcı düzeyi thread / fiber / eşyordam** (coroutine) ise kullanıcı alanında, işbirlikçi olarak (kendi isteğiyle `yield` ederek) değişir: oluşturma ve geçiş çok ucuzdur, yüz binlercesi tutulabilir. Bedeli: bir fiber bloklayan bir sistem çağrısı yaparsa (dosya okuma) onu taşıyan çekirdek thread'i ve dolayısıyla üzerindeki tüm fiber'ler durur; çok çekirdeği kullanmak için M:N eşleme (Go'nun goroutine'leri, Windows fiber'leri, C++20 eşyordamları + iş parçacığı havuzu) gerekir. E4 ve E5'teki zamanlayıcımız tam olarak bir fiber zamanlayıcısıdır: thread'ler `yield` ettikleri yerde kesilir.
:::

:::uyarı Çok thread'li kodda hata ayıklama
Yarış hataları zamanlamaya bağlıdır: hata ayıklayıcı ya da `print` zamanlamayı değiştirip hatayı gizleyebilir ("Heisenbug"). Yararlı alışkanlıklar: paylaşılan durumu en aza indir (thread'e özel kopya + birleştirme); kilitleri her zaman aynı sırada ve RAII ile al (`lock_guard`, `scoped_lock`); bekleme her zaman koşul döngüsünde; testleri çok kez ve farklı yüklerle çalıştır; dinamik analiz araçları kullan (ThreadSanitizer: `-fsanitize=thread`, Helgrind). E4/E5'teki gibi rastgele zamanlama keşfi, gerçek araçların (ör. CHESS, Loom) basit bir hâlidir.
:::
