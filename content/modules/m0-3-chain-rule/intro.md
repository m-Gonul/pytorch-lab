Bir sinir ağı, basit işlemlerden oluşan uzun bir bileşik fonksiyondur. [[zincir kuralı|Zincir kuralı]], bu bileşiğin türevini parçaların yerel türevlerinden kurmayı sağlar. İfadeyi bir [[hesap grafiği]] olarak çizersek:

$$
x,\;y \;\longrightarrow\; a = x+y \;\longrightarrow\; f = a\cdot y
$$

**İleri geçiş** düğüm değerlerini soldan sağa hesaplar. **Geri geçiş** sağdan sola giderken her düğümde "yukarıdan gelen gradyan × yerel türev" çarpımını yapar. Bir değişken birden fazla düğüme giriyorsa, o değişkene gelen katkılar **toplanır**. Bütün backpropagation bu iki kuraldan ibaret.

Vektör değerli $g:\mathbb{R}^n\to\mathbb{R}^m$ için türevlerin tamamı bir matristir, [[jacobian|Jacobian]]: $J_{ij}=\partial g_i/\partial x_j$, shape'i $(m\times n)$.
