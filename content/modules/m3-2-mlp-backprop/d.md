Aynı ağ PyTorch'ta `nn.Sequential(nn.Linear, nn.ReLU, nn.Linear)`. İki görev:

- `two_layer_torch(X, y, W1, b1, W2, b2, reg)` → skaler kayıp tensörü (`F.cross_entropy` serbest). Parametreler `requires_grad=True` geliyor. Kontrol, autograd gradyanlarının c'deki formüllerle aynı olduğunu doğrular.
- `train_mlp(X, y, H, C, epochs=300, lr=0.05)` → `nn.Module`; `Adam`, tam batch.
