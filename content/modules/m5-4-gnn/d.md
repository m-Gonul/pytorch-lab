- `gcn_propagate(x, edge_index, N)` — c'deki `propagate`'in torch hâli (`index_add_`); gradyan akmalı.
- `GCN(d_in, hidden, d_out, dropout=0.5)` — `lin1`, `lin2` (`nn.Linear`); `forward(x, edge_index)`: $\hat A\,\text{lin1}(x)$ → ReLU → dropout → $\hat A\,\text{lin2}(h)$ (logit).
- `train_node_clf(model, X, ei, y, train_mask, steps=200, lr=0.01)` — Adam (`weight_decay=5e-4`), kayıp yalnız `train_mask` düğümlerinde.

Kontrol 3 topluluklu bir stokastik blok modelinde sınıf başına **5 etiketle** eğitir ve aynı özelliklerle bir MLP'ye karşı karşılaştırır.
