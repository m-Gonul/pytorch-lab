NumPy ViBe (gri seviye, `int16` örnekler). Rastgelelik `rng = np.random.default_rng(seed)`; çağrı sırası serbest (kontrol istatistiksel).

- `vibe_init(frame, N=20, seed=0)` — `(N, H, W)`: her örnek, pikselin 3 × 3 komşuluğundan rastgele bir değer (`edge` dolgu; merkez de seçilebilir).
- `vibe_step(samples, frame, rng, R=20, nmin=2, phi=16)` — sınıflandır (`(np.abs(samples − frame) < R).sum(0) ≥ nmin`), `fg` maskesini döndür; **yerinde** güncelle: arka plan piksellerinin `1/φ`'sinde rastgele bir örnek ← piksel değeri; yine arka plan piksellerinin `1/φ`'sinde rastgele bir 8-komşunun (görüntüye kırpılmış) rastgele örneği ← piksel değeri. Gelişmiş indeksleme ile, piksel döngüsü yok.
- `vibe_run(frames, **kw)` — tüm video için ön plan maskeleri `(T, H, W)` (ilk kare başlatma, maskesi 0). Kare döngüsü serbest.
