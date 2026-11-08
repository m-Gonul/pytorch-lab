Çıkarım mühendisliği (eğitilmemiş bir U-Net üzerinde süre — süre ağırlıklardan bağımsızdır). Verilen: d'deki U-Net'in bir kopyası.

- `cfg_two_calls(net, x, t, y, s)` ve `cfg_batched(net, x, t, y, s)` — aynı $\hat\varepsilon$; toplu sürüm koşullu ve koşulsuz girdileri tek batch'te birleştirir.
- `ddpm_sample(net, labels, s, seed)` — 200 adımlı stokastik DDPM (σ² = β̃).
- `ddim_sample(net, labels, s, steps, seed)` — d'deki gibi.
