<!-- çözüm -->

İlk Adam adımı ilginç: bias düzeltmesinden sonra $\hat m_1/\sqrt{\hat v_1}=g_1/|g_1|=-1$, yani ilk adım gradyanın büyüklüğünden bağımsız olarak tam $\eta$. Adam'ın adım büyüklüğü yaklaşık $\eta$ ile sınırlıdır; bu, lr seçimini kolaylaştırır.

İkinci adımda da $\approx\eta$: gradyan yönü değişmediği sürece Adam sabit hızla ilerler. Düz GD'nin aksine minimuma yaklaşırken (gradyan küçülürken) kendiliğinden yavaşlamaz; bu yüzden pratikte lr schedule'ı ile birlikte kullanılır.
