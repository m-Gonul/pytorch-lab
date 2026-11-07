Saf Python. Graf: `{ad: (işlem, [girdi_adları])}`; işlemler `"leaf"`, `"add"`, `"mul"`, `"square"`.

- `forward(graph, inputs)` → her düğümün değeri (sözlük).
- `toposort(graph, out)` → `out` ile biten liste; her düğüm girdilerinden sonra gelir (DFS post-order).
- `backward(graph, vals, out, hooks=None)` → **yalnız yapraklar** için gradyan sözlüğü. Ters topolojik sırada gez; her düğümün toplam gradyanına (varsa) `hooks[ad](g)` uygula, sonra girdilere dağıt.
