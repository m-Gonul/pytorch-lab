CS231n arayüzü, NumPy, **döngü yok**:

- `batchnorm_forward(x, gamma, beta, bn_param)` → `(out, cache)`. `bn_param`: `mode` (`"train"`/`"test"`), `eps` (varsayılan 1e-5), `momentum` (varsayılan 0.9), `running_mean`, `running_var` (yoksa sıfır/bir ile oluştur). Eğitimde $\text{running}=m\cdot\text{running}+(1-m)\cdot\text{batch}$ (CS231n konvansiyonu, yanlı varyans) ve `bn_param`'ı **yerinde** güncelle; testte running istatistiklerle normalize et.
- `batchnorm_backward(dout, cache)` → `(dx, dgamma, dbeta)`
- `layernorm_forward(x, gamma, beta, eps=1e-5)` → `(out, cache)`; istatistik her **satırda**.
- `layernorm_backward(dout, cache)` → `(dx, dgamma, dbeta)`
