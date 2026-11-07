Grafiği çiz: $a$ ve $b$ bir çarpma düğümüne, sonucu $e$ ile $c$ bir toplama düğümüne, sonucu $d$ ile $f$ son çarpma düğümüne giriyor.

Geri geçişte iki kuralı uygula: **toplama** düğümü gelen gradyanı değiştirmeden dağıtır; **çarpma** düğümü her girdiye "gelen gradyan × diğer girdinin değeri" verir. Bu iki kural, sinir ağlarındaki gradyan akışının sezgisinin yarısıdır.

İkinci grupta tek bir tanh nöronu var. $w_2$'nin gradyanının neden 0 olduğunu düşün: girdisi $x_2=0$ olan bir ağırlık o örnekten hiçbir şey öğrenemez.
