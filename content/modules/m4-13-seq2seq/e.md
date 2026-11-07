`step(tokens (N,), h (N, d)) → (log olasılıklar (N, V), yeni h)` — rastgele başlatılmış küçük bir GRU dil modeli. Kurallar (eşitlik testi için): tam `T` adım; EOS üretmiş hipotez yalnız EOS ile, log olasılık 0 ekleyerek uzar; başlangıçta tek hipotez; sonuç en yüksek toplam log olasılık.

- `beam_loop(step, h0, sos, eos, k, T)` — her batch elemanı ve her hipotez için ayrı `step` çağrısı.
- `beam_batched(step, h0, sos, eos, k, T)` — `(B·k)` hipotezi tek `step` çağrısıyla; aday skorları `(B, k·V)` → `topk`; geri izleme indeksleri `// V`, `% V`. Başlangıçta k − 1 hipotezin skoru −∞.
- İkisi de `(B, T)` token ve `(B,)` skor döndürür.
