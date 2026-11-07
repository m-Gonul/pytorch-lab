Önce ileri geçiş: iki gizli nöronun ikisi de pozitif, ReLU ikisini de geçiriyor. Çıktı 3, hedef 1.

Geri geçişte gradyanı çıktıdan geriye doğru taşı. $dh$ iki gizli nörona ulaştığında ReLU maskesi her ikisini de geçiriyor ($z>0$). Bir nöronun $z$'si negatif olsaydı o nörona hiç gradyan akmazdı ("ölü ReLU").

Son olarak $dx$: girdiye göre gradyan, adversarial örneklerin ve saliency haritalarının temeli.
