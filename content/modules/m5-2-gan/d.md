Hedef dağılım: yarıçapı 2 olan çember üzerinde 8 Gauss (σ = 0.05). Mod çöküşünü görmek için klasik deney.

- `Generator(z_dim=2, hidden=128)` ve `Discriminator(hidden=128)` — `Linear → ReLU → Linear → ReLU → Linear`; ayırıcı **logit** döndürür.
- `d_loss(D, real, fake)` — BCE-with-logits: gerçek → 1, sahte → 0. Sahte örnekler buradan $G$'ye gradyan **taşımamalı** (`detach`).
- `g_loss(D, fake)` — non-saturating: sahte → 1 etiketiyle BCE.
- `train_gan(G, D, sample_real, steps=3000, batch=256, lr=1e-3)` — Adam (β₁ = 0.5); her adımda bir D, bir G güncellemesi.

Kontrol 2000 örnek üretip kaç modun kapsandığını ve örneklerin ne kadarının bir moda 0.2'den yakın olduğunu ölçer.
