Üç araç, üç farklı mekanizma:

**Dropout.** Eğitimde her aktivasyonu $p$ olasılıkla sıfırla. **Inverted** sürümde kalanlar $1/(1-p)$ ile ölçeklenir; böylece $\mathbb E[y]=x$ olur ve test zamanında hiçbir şey yapılmaz:

$$
m_i\sim\text{Bernoulli}(1-p),\qquad y=\frac{x\odot m}{1-p},\qquad \frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\odot\frac{m}{1-p}.
$$

**Weight decay.** Ağırlıkları her adımda sıfıra doğru çekmek (M1.3, M3.4): $w\leftarrow w(1-\eta\lambda)$.

**Early stopping.** Doğrulama kaybını her epoch izle; `patience` epoch boyunca iyileşme yoksa dur ve **en iyi** epoch'un ağırlıklarını geri yükle. Eğitim süresini bir hiperparametreye çevirir.

Dropout'un bir yan ürünü: test zamanında da açık bırakıp $T$ kez tahmin almak (**MC dropout**) modelin belirsizliğine dair bir tahmin verir: ortalama tahmin, tahminlerin std'si, ortalama dağılımın entropisi.
