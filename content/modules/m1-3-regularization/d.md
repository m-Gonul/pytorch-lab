PyTorch'ta gradyanla aynı amaç: $\text{MSE}+\lambda_2\lVert w\rVert^2+\lambda_1\lVert w\rVert_1$.

- `fit_reg(X, y, lam2, lam1=0.0, steps=3000, lr=0.05)` → `nn.Linear`; cezayı kayba **elle** ekle (yalnız `weight`'e, `bias`'a değil), `optim.SGD`. $\lambda_1=0$ iken sonuç ridge kapalı formuna eşit olmalı: $(\tfrac1n X_c^\top X_c+\lambda_2 I)w=\tfrac1n X_c^\top y_c$.
- `fit_weight_decay(X, y, lam2, steps=3000, lr=0.05)` — cezayı kayba **eklemeden**, `optim.SGD`'nin `weight_decay` parametresiyle aynı modeli elde et. İki şeye dikkat: doğru `weight_decay` değeri ve bias'ın decay'den muaf tutulması (param groups: `[{"params": [w], "weight_decay": …}, {"params": [b], "weight_decay": 0}]`).
