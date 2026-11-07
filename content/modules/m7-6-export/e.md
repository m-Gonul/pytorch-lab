AOTInductor: ExportedProgram'ı derlenmiş bir pakete çevir ve Python'dan yükle.

- `compile_package(ep, path)` → paket yolu (`torch._inductor.aoti_compile_and_package(ep, package_path=path)`).
- `load_package(path)` → çağrılabilir (`torch._inductor.aoti_load_package`).

İlk derleme CPU'da ~20–60 sn sürer (C++ derleyicisi gerekir). Kontrol farklı batch boyutlarında çıktıyı ve batch 1 gecikmesini eager ile karşılaştırır.
