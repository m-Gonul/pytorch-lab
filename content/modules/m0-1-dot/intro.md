İki vektörün [[shape]]'i aynıysa iç çarpımları tek bir sayıdır:

$$
u\cdot v=\sum_{i=1}^{d}u_i v_i = \lVert u\rVert\,\lVert v\rVert\cos\theta .
$$

Bu iki tanım arasındaki köprü ML'in her yerinde: bir nöronun ön-aktivasyonu $w\cdot x$, bir [[attention]] skoru $q\cdot k$, bir embedding araması kosinüs benzerliği. Norm, vektörün kendisiyle iç çarpımının kökü: $\lVert u\rVert=\sqrt{u\cdot u}$.

Bu modülde aynı hesabı beş kez yazacaksın. Her seferinde bir öncekinin sonucuyla eşleşmesi gerekiyor; değişen tek şey hesabı **kimin** ve **kaç eleman için birden** yaptığı.
