Bir [[convolution]] katmanı, $F$ adet $C\times K_h\times K_w$ çekirdeği girdi üzerinde kaydırır; her konumda pencere ile çekirdeğin iç çarpımı bir çıktı pikseli verir:

$$
Y_{n,f,i,j}=b_f+\sum_{c=1}^{C}\sum_{a=0}^{K_h-1}\sum_{b=0}^{K_w-1}W_{f,c,a,b}\;X^{\text{pad}}_{n,c,\,iS+a,\,jS+b},
\qquad
H_{\text{out}}=\Big\lfloor\frac{H+2P-K_h}{S}\Big\rfloor+1 .
$$

Doğrudan uygulama altı iç içe döngüdür. [[im2col]] bunu tek matris çarpımına çevirir: her pencereyi bir sütuna aç ($C K_hK_w\times H_{\text{out}}W_{\text{out}}$ matris), çekirdekleri satırlara diz ($F\times CK_hK_w$), çarp:

$$
Y_{\text{mat}}=W_{\text{mat}}\cdot\text{im2col}(X)+b .
$$

Geri geçişte $dW=dY_{\text{mat}}\,\text{im2col}(X)^\top$; girdi gradyanı ise $W_{\text{mat}}^\top dY_{\text{mat}}$'yi pencere konumlarına geri dağıtan **col2im** (örtüşen pencerelerde toplanır). PyTorch'ta im2col `F.unfold`, col2im `F.fold`.

:::sezgi Görsel korteksten gelen fikir
Hubel ve Wiesel (1959, 1962) kedinin birincil görsel korteksinde (V1) her nöronun görüş alanının yalnız küçük bir bölgesine (**alıcı alan**) tepki verdiğini, "basit" hücrelerin belirli yönelimli kenarlara, "karmaşık" hücrelerin ise aynı kenara biraz farklı konumlarda da tepki verdiğini buldu. Görme yolunda (V1 → V2 → V4 → IT) alıcı alanlar büyür, öznitelikler karmaşıklaşır. Konvolüsyon ağı bu üç gözlemi mimariye çevirir: **yerel bağlantı** (küçük çekirdek = alıcı alan), **ağırlık paylaşımı** (aynı kenar dedektörü her yerde: öteleme eşdeğerliği), **havuzlama** (karmaşık hücre gibi küçük kaymalara dayanıklılık) ve katmanlar boyunca büyüyen alıcı alan. Fukushima'nın Neocognitron'u (1980) bu fikrin ilk ağıydı; LeCun (1989) gradyanla eğitti. Benzetme esin kaynağıdır, model değil: gerçek korteks geri besleme bağlantıları ve ağırlık paylaşmayan nöronlarla çalışır.
:::

Aşağıdaki deneyde çekirdek, adım, dolgu ve genişlemenin çıktıyı ve boyunu nasıl değiştirdiğini gör.
