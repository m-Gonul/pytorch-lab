NumPy ile ONNX benzeri mini çalışma zamanı. Düğüm: `{"op", "inputs", "output", …}`; işlemler `Gemm` ($xW^\top+b$), `Relu`, `Add`, `Mul`, `Transpose`, `Softmax` (son eksen). `inits`: sabit tensörler (ağırlıklar).

- `run_graph(nodes, inits, feeds)` → tüm değerler sözlüğü.
- `fold_constants(nodes, inits)` → `(yeni_düğümler, yeni_inits)`: tüm girdileri `inits`'te olan düğümleri hesapla, çıktılarını `inits`'e taşı, düğümü sil.
- `infer_shapes(nodes, inits, input_shapes)` → `{ad: şekil}`; şekiller tuple, sembolik boyutlar dize (`"N"`). Yayınlama (`Add`, `Mul`) için: eşitse o, biri 1 ise diğeri.
