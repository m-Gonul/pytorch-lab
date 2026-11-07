Hepsi aynı iskelet: gradyandan bir **durum** güncelle, durumdan bir adım hesapla. PyTorch'un formülleriyle ($g_t$ adım $t$'deki gradyan):

| yöntem | durum | adım |
|---|---|---|
| SGD + momentum | $u\leftarrow\beta u+g$ | $w\leftarrow w-\eta u$ |
| RMSProp | $s\leftarrow\alpha s+(1-\alpha)g^2$ | $w\leftarrow w-\eta\,g/(\sqrt s+\epsilon)$ |
| Adam | $m\leftarrow\beta_1m+(1-\beta_1)g$, $v\leftarrow\beta_2v+(1-\beta_2)g^2$ | $w\leftarrow w-\eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}$ |
| AdamW | Adam ile aynı | önce $w\leftarrow w(1-\eta\lambda)$, sonra Adam |

Adam'ın [[bias düzeltmesi]]: $\hat m=m/(1-\beta_1^t)$, $\hat v=v/(1-\beta_2^t)$. Sıfırdan başlayan hareketli ortalamaların ilk adımlardaki küçüklüğünü telafi eder.

RMSProp ve Adam her parametreyi kendi gradyan geçmişinin büyüklüğüyle ölçekler: sık ve büyük gradyan alan parametreler küçük, nadir gradyan alanlar büyük adımlar atar. Adam'da (L2) `weight_decay` gradyana $\lambda w$ ekler; AdamW ise decay'i ölçekleme dışında, doğrudan parametreye uygular.
