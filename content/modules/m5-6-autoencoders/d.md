PyTorch. Veri: sklearn rakamları (64 piksel, [0, 1]); 497 test, 1300 havuz, havuzdan 50 etiketli.

- `AE(sizes=(64, 32, 12), tied=False)` — kodlayıcı ReLU (kod katmanı doğrusal), kod çözücü aynalanmış ReLU, çıkış sigmoid; `tied=True`: kod çözücü `F.linear(h, W_enc.T, b_dec)`. `encode`, `decode`, `forward`.
- `train_ae(model, X, epochs=100, lr=3e-3, mask=0.0)` — MSE; `mask > 0`: maskeleme gürültüsü (gürültü giderici).
- `greedy_pretrain(sizes, X, epochs)` — her aşamada sığ bir AE eğit, kodla, bir sonrakine geç; sonunda ağırlıkları bir `AE(sizes)`'a kopyala.
- `classify(encoder_ae, X, y, Xt, yt, freeze, steps=300)` — `encoder_ae` None ise sıfırdan; ReLU(kod) → doğrusal 10 sınıf.
