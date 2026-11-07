**Toplu çapraz doğrulama:** k kat × c kapasite modeli, `torch.bmm` ile tek optimizasyonda.

- `batched_cv(widths, k=5, epochs=60, seed=0)` → `{genişlik: (ortalama doğrulama doğruluğu, std)}` — tek gizli katmanlı MLP'ler (genişlik farklı → en büyük genişlikte tensör, fazlası maskeli), her kat kendi eğitim/doğrulama indeksleri.
- `loop_cv(widths, k=5, epochs=60, seed=0)` — aynısı döngüyle.
