- `QuantLinear` — `from_float(linear)` sınıf metodu: kanal başına simetrik int8; `weight_int8` (int8) ve `scale` (fp32, `(out,)`) **buffer** olarak, `bias` kopyası. `forward(x)`: `F.linear(x, weight_int8.float() * scale[:, None], bias)` (ağırlık-yalnız kuantizasyon: depolama int8, hesap fp32).
- `quantize_model(model)` — tüm `nn.Linear`'ları (iç içe dahil) `QuantLinear` ile değiştir; yeni model döndür (`copy.deepcopy`).
- `state_bytes(model)` — `state_dict` tensörlerinin bayt toplamı.
- `prune_global(model, amount)` — tüm `nn.Linear` ağırlıklarında `prune.global_unstructured(…, L1Unstructured, amount)`, sonra `prune.remove` (maskeyi kalıcı yap).

Not: `torch.ao.quantization` PyTorch'ta kullanımdan kaldırılıyor; yerini `torchao` alıyor. Burada mekanizmayı görmek için kendi modülünü yazıyorsun.
