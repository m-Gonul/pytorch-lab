<!-- çözüm -->

İleri geçişte tek girdi ilk adımda; sonrasında durum her adımda yaklaşık yarıya iniyor ($w=0.5$).

Geri geçişte gradyan her adımda $w(1-h^2)<0.5$ ile çarpılıyor. Üç adımda $0.21$'e düştü; 30 adımda $10^{-10}$ civarında olurdu. RNN'in uzak geçmişi "unutması" bu çarpımdan.

$\partial L/\partial u$'da $u$ üç adımda da kullanıldı ama $x_2=x_3=0$ olduğu için yalnız ilk adımın katkısı kaldı.
