Saf Python, kategorik Naive Bayes. Satırlar tuple, etiketler liste. `math` ve `collections` serbest.

- `fit_nb(rows, labels, alpha=1.0)` → bir model (yapısı senin seçimin). Laplace paydasındaki $V_j$, $j$. özelliğin eğitimde görülen **farklı** değer sayısı.
- `predict_proba(model, row)` → `{sınıf: olasılık}`; skorları **log uzayında** topla, en büyük log'u çıkararak normalize et.
- `predict(model, row)` → en olası sınıf.

Kontrol, `alpha=0` ile elle bulduğun 0.4'ü, `alpha=1` ile 0.4286'yı ve görülmemiş bir değerin sıfır olasılık üretmesini test eder.
