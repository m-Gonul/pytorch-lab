NumPy; graf formatı b'deki gibi `(ad, işlem, argümanlar)` listesi, işlemler: `"input"`, `"add"`, `"mul"`, `"tanh"`, `"exp"` (elementwise) ve `"sum"` (indirgeme, füzyon sınırı).

- `evaluate(graph, inputs)` → her düğümün değeri (sözlük).
- `fuse(graph, outputs)` → gruplar listesi (düğüm adı listeleri, `input`'lar hariç). Elementwise bir düğüm, elementwise girdisinin grubuna katılır **eğer** o girdinin tek tüketicisi bu düğümse ve girdi bir çıkış değilse. `sum` kendi grubunda.
- `bytes_moved(graph, groups, nbytes, outputs)` — her grup, dışarıdan okuduğu değerlerin baytını okur ve grubun dışında kullanılan (ya da çıkış olan) değerlerinin baytını yazar. `nbytes[ad]` değer başına bayt.
