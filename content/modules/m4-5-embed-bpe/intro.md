Bir dil modeli metni önce **token** dizisine çevirir. [[bpe|Byte-pair encoding]]:

1. Her kelimeyi karakterlerine böl (GPT-2'de baytlarına).
2. Tüm komşu sembol çiftlerini frekansla say.
3. En sık çifti yeni bir sembole birleştir; birleştirme listesine ekle.
4. İstenen sözlük boyutuna kadar 2–3'ü tekrarla.

Kodlama: yeni bir kelimeyi karakterlere böl ve öğrenilen birleştirmeleri **aynı sırayla** uygula.

Her token id'si bir **embedding** vektörüne eşlenir: $E\in\mathbb R^{V\times d}$, $e=E[\text{id}]$. Bu, one-hot vektörle matris çarpımıyla aynıdır, $e=\text{onehot}(\text{id})^\top E$, ama $V$ kat daha ucuzdur. Gradyan ise kullanılan satırlara **toplanarak** dağılır:

$$
\frac{\partial L}{\partial E_v}=\sum_{t:\ \text{id}_t=v}\frac{\partial L}{\partial e_t}\qquad(\text{scatter-add}).
$$
