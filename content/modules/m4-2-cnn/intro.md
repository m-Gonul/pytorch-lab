CNN tarihinin üç fikri:

- **LeNet → VGG:** küçük çekirdekleri üst üste koymak. İki 3×3 katman bir 5×5'in [[receptive field]]'ını görür ama $2\cdot9C^2=18C^2$ parametreyle ($25C^2$ yerine) ve araya bir doğrusal olmayanlık daha ekleyerek.
- **Receptive field** katman katman büyür: $r\leftarrow r+(k-1)\,j$, $j\leftarrow j\cdot s$ ($j$: o ana kadarki toplam stride).
- **ResNet:** her blok $y=\text{ReLU}\big(x+F(x)\big)$. Derinlik arttıkça düz ağların *eğitim* hatası bile artıyordu (optimizasyon sorunu). [[residual bağlantı]] Jacobian'ı $I+J_F$ yapar; gradyan kimlik yolundan sönmeden geçer.

BasicBlock: `conv3×3 → BN → ReLU → conv3×3 → BN`, artı shortcut. Çözünürlük ya da kanal sayısı değişen blokta shortcut bir `conv1×1 (stride) → BN` projeksiyonudur.

Çıkarımda (eval) BatchNorm sabit bir afin dönüşümdür ve önceki conv'a katlanabilir:

$$
W'=W\cdot\frac{\gamma}{\sqrt{\sigma^2+\epsilon}},\qquad b'=(b-\mu)\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}+\beta .
$$
