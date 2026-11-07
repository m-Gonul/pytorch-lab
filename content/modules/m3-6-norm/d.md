PyTorch modülleri, öğrenilen parametreler `nn.Parameter`, öğrenilmeyen durum `register_buffer`:

- `MyBatchNorm1d(num_features, eps=1e-5, momentum=0.1)` — `nn.BatchNorm1d` ile aynı: eğitimde batch istatistiği (yanlı varyansla normalize), running güncellemesi $\text{run}=(1-m)\,\text{run}+m\cdot\text{batch}$, running_var için **yansız** batch varyansı; `eval()`'de running. `forward` yalnız `(N, C)` girdi.
- `MyRMSNorm(dim, eps=1e-6)` — `nn.RMSNorm(dim, eps=eps)` ile aynı: $y=x/\sqrt{\text{mean}(x^2)+\epsilon}\odot w$, son eksende.

Kontrol ileri çıktıları, gradyanları ve 5 eğitim adımı sonrası running istatistikleri karşılaştırır. Autograd'ı kullanabilirsin; elle backward gerekmiyor.
