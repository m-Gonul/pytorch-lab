Saf Python, torch.fx'in minik bir kopyası. `Graph` hazır: `nodes` = `(ad, işlem, argümanlar)` listesi, `add(op, args)` yeni düğüm adı döndürür, `output` çıkış düğümünün adı. İşlemler: `"input"`, `"add"`, `"sub"`, `"mul"`, `"tanh"`, `"exp"`. Argümanlar düğüm adları (`str`) ya da sabit sayılar.

- `Proxy(graph, name)` — `+`, `-`, `*` (sağdan sayılarla da) yeni düğüm kaydeder ve yeni `Proxy` döndürür.
- `tanh(v)`, `exp(v)` — `Proxy` ise düğüm kaydet, sayıysa `math` ile hesapla.
- `trace(fn, n_inputs)` → `Graph`.
- `run(graph, inputs)` → çıkış değeri.
- `cse(graph)` — aynı `(işlem, argümanlar)` düğümlerini tek düğümde birleştir (`add` ve `mul` değişmeli: argüman sırası önemsiz), kullanımları yeniden yönlendir.
- `dce(graph)` — çıkışa ulaşmayan düğümleri sil (girdiler kalır).
