Eğitimde GPU belleği dört kaleme gider:

| kalem | fp32 Adam, parametre başı |
|---|---|
| ağırlıklar | 4 bayt |
| gradyanlar | 4 bayt |
| Adam $m$, $v$ | 8 bayt |
| aktivasyonlar | batch, dizi uzunluğu ve derinlikle **ölçeklenir** |

Aktivasyonlar, geri geçiş için saklanan ara tensörler (M6.1). Büyük batch'lerde ve uzun dizilerde baskın kalem genelde budur.

**[[activation checkpointing]]:** ileri geçişte yalnız bazı katmanların çıktısını sakla; geri geçişte bir segmente gelince, onun ara çıktılarını segment başından **yeniden hesapla**. $n$ katmanlı zincirde her $k$ katmanda bir saklarsan tepe bellek $\approx n/k+k$, ek hesap $\approx$ bir ileri geçiş.

**Gradient accumulation:** bellek tek seferde 64 örneğe yetmiyorsa, 4 mikro-batch × 16 ile gradyanları biriktir ve bir kez adım at. Ortalama kayıp için her mikro-batch kaybını mikro-batch sayısına böl.
