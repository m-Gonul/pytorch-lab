Kosinüsün $u$'ya göre gradyanı, embedding'leri eğitirken (ör. contrastive loss) ağın gördüğü sinyaldir. Bölüm kuralıyla:

$$
\frac{\partial \cos\theta}{\partial u}=\frac{v}{\lVert u\rVert\lVert v\rVert}-\cos\theta\,\frac{u}{\lVert u\rVert^2}.
$$

İkinci terim gradyanın $u$ yönündeki bileşenini siler: uzunluğu değiştirmek kosinüsü değiştirmez, yani gradyan $u$'ya **diktir**. Kontrol bunu da doğrular.

- `cosine_torch(u, v)` — tensörlerle kosinüs.
- `cosine_grad(u, v)` — [[autograd]] ile $\partial\cos\theta/\partial u$.
