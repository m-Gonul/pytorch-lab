Triton (GPU yoksa yorumlayıcı).

- `copy(x)` — 1B blok kopyası (maske ile).
- `transpose_naive(x)` — her program bir satır parçası okur (bitişik), devrik konuma (adımlı) yazar.
- `transpose_tiled(x, BM=16, BN=16)` — 2B program ızgarası; `[BM, BN]` blok yükle, `tl.trans`, devrik bloğa yaz; kenarlarda maske.

Not: Triton paylaşımlı belleği ve birleştirmeyi kendisi yönetir; burada çekirdeğin hangi adresleri ürettiğini c'deki modelle düşünüyoruz.
