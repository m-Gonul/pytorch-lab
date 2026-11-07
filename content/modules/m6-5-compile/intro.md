`torch.compile(fn)` üç katmandan oluşur:

1. **Dynamo** — Python bytecode'unu çalışma anında yakalar, tensör işlemlerini bir FX grafına koyar. Yakalayamadığı (ör. veriye bağlı Python akışı, `print`, `.item()`) yerde [[graph break]] olur: graf ikiye bölünür, araya normal Python girer.
2. **AOTAutograd** — ileri ve geri grafını önceden çıkarır.
3. **Inductor** — grafı optimize eder, elementwise işlemleri [[operatör füzyonu]]yla birleştirir, GPU'da Triton, CPU'da C++ kodu üretir.

Elementwise işlemler **bellek-bağımlıdır**: bir `add`, okuduğu her 8 bayt için 1 FLOP yapar. Üç işlemi zincirlemek her ara sonucu belleğe yazıp geri okumak demek; füzyonla girdi bir kez okunur, çıktı bir kez yazılır.

**torch.fx** grafı Python'dan erişilebilir kılar: `symbolic_trace(model)` her işlemi bir `Node` (`placeholder`, `call_module`, `call_function`, `output`) olarak kaydeder; düğümleri değiştirip `GraphModule` olarak yeniden derleyebilirsin.
