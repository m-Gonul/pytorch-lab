`torch.func` (eski adıyla functorch) türevi **fonksiyon dönüşümü** olarak sunar:

| dönüşüm | ne verir | maliyet |
|---|---|---|
| `grad(f)` | $\nabla f$ (skaler $f$) | 1 geri geçiş |
| `vjp(f, x)` | $u\mapsto u^\top J$ | 1 geri geçiş |
| `jvp(f, (x,), (v,))` | $Jv$ ([[jvp]]) | 1 ileri geçiş (dual sayılar) |
| `jacrev` / `jacfwd` | $J$ | çıktı sayısı / girdi sayısı kadar |
| `vmap(f)` | $f$'yi batch boyutu üzerinde | tek batch'li çağrı |

Bunlar bileşebilir: `vmap(grad(f))` örnek başı gradyan, `jacfwd(jacrev(f))` Hessian, `jvp(grad(f), …)` Hessian-vektör çarpımı.

Bir `nn.Module`'ü parametrelerinin saf fonksiyonu olarak çağırmak için `torch.func.functional_call(model, params, (x,))`.

**Forward-mode** dual sayılarla çalışır: her değer $(a, \dot a)$ çifti, kurallar $(a,\dot a)\cdot(b,\dot b)=(ab,\ \dot ab+a\dot b)$ gibi.
