Saf Python.

- `to_json(layers)` / `from_json(s)` — `layers`: `{"op": "linear", "W": [[…]], "b": […]}` ya da `{"op": "relu"}` sözlükleri; `json` modülüyle.
- `run(layers, x)` — `x` liste; linear: $Wx+b$.
- Kayıt tabanlı izleyici: `Rec(values, graph)` — `*` (sayıyla), unary `-` işlemleri `graph`'a `("mul", c)` / `("neg",)` ekler ve yeni `Rec` döndürür; `sum()` **somut bir sayı** döndürür (kayıt yok).
- `trace(fn, x)` → graf listesi; `replay(graph, x)` → liste.
