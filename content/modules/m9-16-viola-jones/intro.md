**Haar benzeri öznitelikler:** bitişik dikdörtgenlerin toplamları arasındaki fark. Beş tür (scikit-image adlarıyla): `type-2-x`, `type-2-y` (kenar), `type-3-x`, `type-3-y` (çizgi), `type-4` (çapraz). İşaretler: iki dikdörtgende $r_2-r_1$, üçte $r_2-(r_1+r_3)$ (orta − kenarlar), dörtte $(r_2+r_4)-(r_1+r_3)$ (dikdörtgenler sol üst, sağ üst, sağ alt, sol alt sırasıyla). Yüzdeki göz bandı yanaklardan koyu, burun köprüsü gözlerden açık: bu karşıtlıkları yakalarlar. 24 × 24 pencerede tüm konum ve boyutlarda **162 336** öznitelik vardır.

**Integral görüntü:** $ii(y,x)=\sum_{y'<y,\,x'<x}f(y',x')$; dikdörtgen toplamı 4 erişim. Pencere başına normalizasyon için ayrıca $f^2$'nin integral görüntüsü: $\sigma^2=E[f^2]-\mu^2$.

**AdaBoost (Viola–Jones biçimi):**

1. Ağırlıklar: pozitiflere $1/2m$, negatiflere $1/2l$.
2. Her tur $t$: ağırlıkları normalize et; her öznitelik için en iyi **karar kütüğü** $h(x)=[p\,f(x)<p\,\theta]$ (eşik $\theta$, polarite $p$) ve ağırlıklı hatası $\epsilon$; en küçük $\epsilon$'lusunu seç.
3. $\beta_t=\epsilon_t/(1-\epsilon_t)$; doğru sınıflananların ağırlığını $\beta_t$ ile çarp.
4. Güçlü sınıflandırıcı: $\sum_t\alpha_th_t(x)\ge\frac12\sum_t\alpha_t$, $\alpha_t=\log1/\beta_t$.

**Kaskad:** aşamalar sırayla; bir aşama "değil" derse pencere hemen reddedilir. Aşama eşikleri tespit oranı ~%99'u koruyacak şekilde düşürülür. 10 aşama, $d_i=0.99$, $f_i=0.3$: $D=0.90$, $F=6\times10^{-6}$. İlk aşamalar 2–10 öznitelikle pencerelerin çoğunu eler.

**Tarama:** pencereyi tüm konum ve ölçeklerde kaydır (görüntüyü küçültmek yerine öznitelikleri büyüt), örtüşen tespitleri birleştir. Artıları: gerçek zamanlı, aydınlatmaya görece dayanıklı. Eksileri: yalnız dik ve önden yüzler, dönmeye değişmez değil; bugün CNN tabanlı dedektörler geçti.
