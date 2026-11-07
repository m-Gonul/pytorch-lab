# pytorch-lab

**Elle → döngü → NumPy → PyTorch → vektörize.** Klasik ML'den ileri PyTorch'a giden, tarayıcıda çalışan Türkçe ders sitesi. Her konu aynı beş aşamadan geçer ve her aşama bir öncekinin sonucuyla birebir eşleşmek zorundadır:

| Aşama | Ne yapılır | Nerede çalışır |
|---|---|---|
| 1 Elle | 2×2 / 3 elemanlı örnekte her ara değer kâğıtta hesaplanıp forma girilir | Sayfa içinde (motor yok) |
| 2 Döngü | Saf Python, `for`/`if`; `numpy`/`torch` yasak | Tarayıcı (Pyodide) |
| 3 NumPy | Döngüsüz; AST denetimiyle `for` yasağı | Tarayıcı (Pyodide) |
| 4 PyTorch | Tensor + autograd; gradyanlar formülle doğrulanır | Yerel motor (`uv run`) ya da Colab |
| 5 Vektörize | Batch, `einsum`, `vmap`; hız tablosu üretilir | Yerel motor ya da Colab |

Site: https://m-gonul.github.io/pytorch-lab/

## Mimari

Sunucu yok; GitHub Pages'te statik site.

- **Tarayıcı motoru** — [Pyodide](https://pyodide.org) (WebAssembly CPython) bir Web Worker'da; NumPy/pandas/scikit-learn tam destekli. PyTorch Pyodide'de yok.
- **Yerel PyTorch motoru** — `runner/torchlab.py`, PEP 723 başlıklı tek dosya. `uv run https://m-gonul.github.io/pytorch-lab/runner/torchlab.py` kendi ortamını (torch, numpy, pandas) kurar, `127.0.0.1`'i dinler, token ve origin kontrolü yapar, torch'u önceden import etmiş bir worker süreci hazır tutar.
- **Colab** — `notebooks/<modül>.ipynb`, içerikten üretilir; aynı kontrolleri `%%writefile` + `labcheck.run_file` ile çalıştırır.
- **Tek kontrol çekirdeği** — `site/py/labcheck.py` üç motorda ve CI'da birebir aynı dosya. Float karşılaştırması `allclose` tabanlı; döngü/import yasakları AST ile denetlenir.

## Dizin yapısı

```
content/
  site.yaml            site ayarları (repo, Pyodide sürümü, runner portu)
  curriculum.yaml      8 faz, 34 modül — müfredatın tek kaynağı
  sources.yaml         kaynak haritası ve lisanslar
  glossary.yaml        terim sözlüğü; metinde [[terim]] → tooltip
  pages/kurulum.md     motorlar sayfası
  modules/<id>/
    module.yaml        başlık, hedefler, kaynaklar, el hesabı alanları + verify betiği, sınama soruları
    intro.md a.md … e.md
    b.py  b.sol.py  b.check.py     (c, d, e için de aynı üçlü)
site/                  statik varlıklar: css, js (motorlar, editör), py/labcheck.py
runner/torchlab.py     yerel PyTorch motoru
scripts/               build.mjs, notebooks.mjs, serve.mjs, md.mjs
notebooks/             üretilmiş Colab notebook'ları (commit'lenir; CI güncelliğini denetler)
test/                  test_modules.py, test_notebooks.py, site.test.mjs
```

## Geliştirme

```bash
npm ci
npm run dev                         # build + http://localhost:8000
uv run runner/torchlab.py --site http://localhost:8000/    # yerel PyTorch motoru

# testler (Python tarafı gerçek torch ister)
uv venv && uv pip install numpy pandas pyyaml
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/python test/test_modules.py
.venv/bin/python test/test_notebooks.py
npm test
```

## Yeni modül eklemek

1. `content/curriculum.yaml`'daki `id` ile `content/modules/<id>/` klasörü aç; `module.yaml` + `intro.md` + `a.md`…`e.md`.
2. Her kod aşaması için üç dosya: `X.py` (başlangıç iskeleti), `X.sol.py` (model çözüm), `X.check.py` (`def check(ns, R)`). Kontrol yardımcıları: `close`, `no_loops`, `max_loops(src, n, func=...)`, `no_imports`, `need`, `ensure`, `bench`, `R.table`.
3. El hesabı alanlarının cevaplarını `hand.verify` betiği bağımsız olarak hesaplamalı.
4. `npm run notebooks`, sonra testler. CI şunu garanti eder: çözüm geçer, iskelet geçmez, el hesabı cevapları verify ile aynı, notebook güncel, iç bağlantılar kırık değil.

YAML çift tırnaklı dizgilerde LaTeX ters bölüleri ikilenir: `"$\\frac{a}{b}$"`.

## Lisans

Kod [MIT](LICENSE). Ders metni [CC BY-SA 4.0](LICENSE-CONTENT.md) (d2l.ai uyarlamaları bunu gerektirir). Uyarlanan ve bağlantı verilen kaynaklar lisanslarıyla birlikte `content/sources.yaml`'da ve her modülün sonunda listelenir.
