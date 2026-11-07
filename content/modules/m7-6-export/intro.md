Eğitimde model Python'da yaşar; dağıtımda genelde **Python'suz**, sabit bir graf olarak çalıştırılmak istenir.

**Grafı yakalamanın iki yolu:**

- *Tracing* (eski `torch.jit.trace`, ONNX'in eski yolu): örnek bir girdiyle çalıştır, yapılan işlemleri kaydet. Kontrol akışı **o girdinin aldığı dalda donar**.
- *`torch.export`* (Dynamo tabanlı): bytecode'u sembolik olarak izler; veriye bağlı kontrol akışında hata verir, şekillere bağlı koşullar için **[[guard]]** ekler. Dinamik boyutlar `torch.export.Dim` ile işaretlenir; veriye bağlı dallar `torch.cond` ile ifade edilir.

Çıktı bir `ExportedProgram`: ATen düzeyinde işlemlerden oluşan bir FX grafı + ağırlıklar. Buradan:

- `torch.export.save` → `.pt2` dosyası,
- `torch.onnx.export(…, dynamo=True)` → ONNX (ONNX Runtime, tarayıcıda ONNX Runtime Web),
- `torch._inductor.aoti_compile_and_package` → C++'tan yüklenebilen derlenmiş paket (AOTInductor).

Çalışma zamanları grafı tek başına optimize eder: **sabit katlama** (yalnız ağırlıklara bağlı alt ifadeleri önceden hesaplamak), **şekil çıkarımı** (bellek planlaması için), işlem füzyonu.
