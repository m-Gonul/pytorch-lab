Şimdi aynı tanımı Python listeleriyle yaz. Matris, satırlardan oluşan bir liste: `[[1, 2], [3, 4]]`.

- `matmul(A, B)` iki matrisi çarpsın ve yeni bir liste döndürsün; girdileri değiştirmesin.
- İç boyutlar uyuşmazsa `ValueError` fırlatsın.
- `numpy` ve `torch` yasak. Üç iç içe döngü: $i$ (satır), $j$ (sütun), $t$ (toplam indisi).

Döngü sırasına dikkat: `for i … for j … for t` en okunaklı olanı; `i, t, j` sırası $B$'yi satır satır gezdiği için daha "önbellek dostu"dur. Burada fark etmez ama (e)'de C seviyesinde bunun büyük etkisi var.
