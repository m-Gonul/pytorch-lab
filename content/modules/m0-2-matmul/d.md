PyTorch'ta `@` aynı işi yapar; fark, [[autograd]]'ın işlemi kaydedip gradyanı hesaplayabilmesi.

$L = \sum_{ij} C_{ij}$, $C = AB$ olsun. Zincir kuralıyla, $G = \partial L/\partial C$ (burada her elemanı 1 olan $n\times m$ matris) için

$$
\frac{\partial L}{\partial A} = G\,B^\top \quad (n\times k),\qquad
\frac{\partial L}{\partial B} = A^\top G \quad (k\times m).
$$

Bu iki formül, her lineer katmanın backward'ının ta kendisi. Yaz:

- `matmul_torch(A, B)` — iki tensörü çarpsın.
- `matmul_grads(A, B)` — `A` ve `B`'nin kopyalarını `requires_grad=True` yapıp $L$'yi hesaplasın, `L.backward()` çağırsın ve `(dA, dB)` döndürsün.

Kontrol, döndürdüğün gradyanları yukarıdaki formülle karşılaştırır.
