**Yarış olasılığı**, NumPy, döngüsüz. İki thread aynı sayaca oku–değiştir–yaz yapar; her birinin başlangıcı $[0,T]$'de düzgün, okuma–yazma arası L. Aralıklar örtüşürse güncelleme kaybolur.

- `lost_update_prob(L, T, n_trials, seed=0)` → deneysel olasılık; analitik $1-(1-L/T)^2$ ile karşılaştır.
- `expected_lost(k, L, T, n_trials, seed=0)` — k thread: her biri tek artış; sonuç = örtüşme bileşeni başına 1 (bir bileşendeki tüm artışlardan yalnız biri kalır… ). Basitleştirme: başlangıçları sırala; ardışık başlangıç farkı < L ise aynı bileşen. Kayıp = k − bileşen sayısı; ortalama.
