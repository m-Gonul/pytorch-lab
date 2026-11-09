### Problem: kararsız gradyanlar {#problem}

M3.2'de BP2'nin bir **çarpanlar zinciri** olduğunu gördük: $\delta^l=\mathrm{diag}(\sigma'(z^l))(W^{l+1})^\top\cdots\delta^L$. İleri geçiş de bir zincirdir: $a^l=\sigma(W^la^{l-1})$. Her katman sinyalin ölçeğini bir çarpanla değiştirir; $L$ katmanda çarpan $L$ kez uygulanır. Çarpan 1'den biraz küçükse sinyal üstel olarak **söner** (vanishing), biraz büyükse **patlar** (exploding). Sönen gradyanlar ilk katmanların hiç öğrenmemesine, patlayanlar sayısal taşmaya ve ıraksamaya yol açar. 2010'a kadar derin ağların eğitilememesinin başlıca nedeni buydu.

Glorot ve Bengio (2010) iki suçlu buldu: o dönemin yaygın kombinasyonu — **sigmoid aktivasyon** ve **standart normal başlatma** $\mathcal N(0,1)$. Sigmoid'in ortalaması 0.5'tir (sıfır merkezli değil), büyük girdilerde doyar (türev ≈ 0) ve türevi hiçbir zaman 1/4'ü aşmaz; $\mathcal N(0,1)$ ağırlıklar ise her katmanda varyansı $n$ kat büyütüp nöronları doyuma iter. Bu modülün iki aracı: **varyansı koruyan başlatma** ve **doymayan aktivasyonlar**.

### Varyans hesabı {#varyans}

:::türetim Bir katmandan sonra varyans
$z_i=\sum_{j=1}^{n_{\text{in}}}w_{ij}x_j$ (bias 0). Varsayımlar: $w_{ij}$'ler bağımsız, ortalaması 0, varyansı $\mathrm{Var}(w)$; $x_j$'ler birbirinden ve ağırlıklardan bağımsız. Bağımsız ve sıfır ortalamalı $w$ ile çarpım terimlerinin ortalaması 0, varyansı $\mathrm{Var}(w)\,\mathbb E[x^2]$ (M0.4: bağımsızlarda $\mathbb E[w^2x^2]=\mathbb E[w^2]\mathbb E[x^2]$). Bağımsız terimlerin varyansları toplanır:
$$\mathrm{Var}(z)=n_{\text{in}}\,\mathrm{Var}(w)\,\mathbb E[x^2].$$
**Dikkat:** $\mathbb E[x^2]$, $\mathrm{Var}(x)$ değil — $x$ sıfır merkezli değilse (sigmoid, ReLU çıktısı) ikisi farklıdır.
:::

**İleri geçişte koruma:** $\mathrm{Var}(z^l)=\mathrm{Var}(z^{l-1})$ isteniyorsa $n_{\text{in}}\mathrm{Var}(w)\cdot\frac{\mathbb E[x^2]}{\mathrm{Var}(z^{l-1})}=1$. Bu oran aktivasyona bağlıdır:

- **Doğrusal / tanh (sıfır civarında doğrusal):** $\mathbb E[x^2]\approx\mathrm{Var}(z)$ → $\mathrm{Var}(w)=\frac1{n_{\text{in}}}$ (**LeCun**).
- **ReLU:** sıfır merkezli simetrik $z$ için $\mathbb E[\mathrm{ReLU}(z)^2]=\frac12\mathbb E[z^2]$ (negatif yarı sıfırlanır) → $\mathrm{Var}(w)=\frac2{n_{\text{in}}}$ (**He / Kaiming**, He ve ark. 2015). 2 çarpanı, ReLU'nun yarıya indirdiği ikinci momenti telafi eder.

**Geri geçişte koruma:** BP2'de $\delta^{l}=(W^{l+1})^\top\delta^{l+1}\odot\sigma'$; aynı hesap bu kez $n_{\text{out}}$ ile: $\mathrm{Var}(\delta^l)=n_{\text{out}}\mathrm{Var}(w)\mathbb E[\sigma'^2]\mathrm{Var}(\delta^{l+1})$. İleri için $n_{\text{in}}\mathrm{Var}(w)\approx1$, geri için $n_{\text{out}}\mathrm{Var}(w)\approx1$ gerekir — $n_{\text{in}}\ne n_{\text{out}}$ ise ikisi birden sağlanamaz.

:::tanım Glorot/Xavier başlatma (Glorot ve Bengio 2010)
Uzlaşma: $\mathrm{fan}_{\text{avg}}=\frac{n_{\text{in}}+n_{\text{out}}}2$ ile $\mathrm{Var}(w)=\frac1{\mathrm{fan}_{\text{avg}}}=\frac2{n_{\text{in}}+n_{\text{out}}}$. Düzgün dağılımla: $w\sim U(-r,r)$, $r=\sqrt{3/\mathrm{fan}_{\text{avg}}}$ ($U(-r,r)$'nin varyansı $r^2/3$).
:::

| başlatma | aktivasyonlar | $\sigma^2=\mathrm{Var}(w)$ | PyTorch |
|---|---|---|---|
| Glorot | yok (doğrusal), tanh, sigmoid, softmax | $1/\mathrm{fan}_{\text{avg}}$ | `xavier_normal_`, `xavier_uniform_` |
| He | ReLU ve türevleri (Leaky, ELU, GELU…) | $2/\mathrm{fan}_{\text{in}}$ | `kaiming_normal_(nonlinearity="relu")` |
| LeCun | SELU | $1/\mathrm{fan}_{\text{in}}$ | `kaiming_normal_(nonlinearity="linear")` ya da `normal_(std=1/√n)` |

**PyTorch'un varsayılanı:** `nn.Linear` ve `nn.Conv` ağırlıkları $U(-\sqrt{1/n_{\text{in}}},\sqrt{1/n_{\text{in}}})$ ile başlatılır (varyans $\frac1{3n_{\text{in}}}$; tarihsel nedenlerle `kaiming_uniform_(a=√5)`). Derin ReLU ağlarında bu biraz küçüktür; çoğu zaman normalizasyon katmanları (M3.6) farkı telafi eder, ama derin ve normalizasyonsuz ağlarda başlatmayı açıkça ayarla.

Aşağıda katman katman aktivasyon ve gradyan ölçeğini izle: başlatmayı ve aktivasyonu değiştir, derinliği artır.

::lab l1

:::uyarı Sıfırla başlatmak
Tüm ağırlıkları 0 (ya da aynı sayı) yapmak simetriyi kırmaz: bir katmandaki tüm nöronlar aynı girdiyi görür, aynı çıktıyı üretir, aynı gradyanı alır ve **sonsuza kadar aynı kalır** — $n$ nöronlu katman tek nöron gibi davranır. Rastgele başlatma simetri kırıcıdır. (Bias'lar 0 başlatılabilir: simetriyi ağırlıklar zaten kırar.)
:::

### Aktivasyon fonksiyonları {#aktivasyon}

**Sigmoid ve tanh.** $\tanh(z)=2\sigma(2z)-1$: aynı şekil, ama sıfır merkezli ve türevi en fazla 1 (sigmoid'de 1/4). Sıfır merkezlilik M3.2'deki BP4 içgörüsü yüzünden önemlidir: bir katmanın tüm girdileri pozitifse (sigmoid çıktısı) bir nöronun tüm ağırlık gradyanları aynı işaretlidir ve güncellemeler zikzak yapar (ders alıştırması 2). İkisi de büyük $|z|$'de doyar. Bugün gizli katmanlarda nadiren kullanılırlar; kapılarda (LSTM, M4.4) ve çıkışta (olasılık) yaşarlar.

**ReLU** $=\max(0,z)$ (Nair ve Hinton 2010; AlexNet 2012). Pozitif tarafta doymaz (türev 1), hesaplaması çok ucuz, gradyanı aktif yollarda olduğu gibi geçirir. Kusuru: **ölü ReLU**'lar. Bir nöronun ön-aktivasyonu tüm eğitim örneklerinde negatif olursa (büyük bir öğrenme oranı adımı bias'ı çok negatife itebilir) türevi hep 0'dır, hiç gradyan almaz ve bir daha dirilmez. Ağın nöronlarının yarısı bu şekilde ölebilir.

**ReLU ailesi:**

| aktivasyon | $z<0$ için | özellik |
|---|---|---|
| Leaky ReLU | $\alpha z$ ($\alpha$ = 0.01; 0.2 sıklıkla daha iyi) | asla tam ölmez |
| RReLU | $\alpha$ eğitimde rastgele, testte ortalama | düzenlileştirici etkisi |
| PReLU | $\alpha$ öğrenilen parametre | büyük veride iyi, küçükte aşırı uyum riski |
| ELU | $\alpha(e^z-1)$ | ortalama sıfıra yakın, $z=0$'da pürüzsüz ($\alpha=1$), ama üs hesabı yavaş |
| SELU | $\lambda\cdot$ELU, $\alpha\approx1.6733$, $\lambda\approx1.0507$ | öz-normalizasyon (aşağıda) |
| GELU | $z\Phi(z)$ | Transformer'ların varsayılanı (BERT, GPT) |
| Swish / SiLU | $z\sigma(z)$ | GELU'ya çok benzer; EfficientNet, LLaMA (SwiGLU içinde) |

GELU ve Swish pürüzsüz, tekdüze olmayan (negatif tarafta küçük bir çukur) aktivasyonlardır; büyük modellerde ReLU'dan biraz daha iyi sonuç verirler.

:::ayrıntı SELU ve öz-normalizasyon (Klambauer ve ark. 2017)
SELU'nun sabitleri öyle seçilmiştir ki, girdiler ortalama 0 / varyans 1 ve ağırlıklar LeCun başlatmalıyken, bir katmanın çıktısı da (yaklaşık) ortalama 0 / varyans 1 olur — ve bu nokta **çekicidir**: biraz saparsa sonraki katmanlar geri çeker. Derin bir yoğun ağ normalizasyon katmanı olmadan kendini normalize eder. **Koşulları:** girdiler standartlaştırılmış; LeCun normal başlatma; ağ yalnız yoğun katmanların sıralı bir yığını (artık bağlantılar, RNN'ler bozar); düzenlileştirme olarak dropout yerine **alpha dropout** (M3.7). Koşullar sağlanmazsa garanti kaybolur.
:::

Aşağıda bir aktivasyon seç, girdinin ortalamasını ve ölçeğini değiştir: çıktının ortalaması, ikinci momenti, türevin ortalaması ve "ölü/doymuş" oran canlı hesaplanır.

::lab l2

**Hangisini kullanmalı?** Genel bir sıralama (ders slaytları, Géron): SELU > ELU > Leaky ReLU (ve türevleri) > ReLU > tanh > sigmoid — ama bu kaba bir sıralamadır. Pratik kural: **önce ReLU** (hızlı, donanım optimizasyonları var, iyi çalışır); Transformer'larda GELU/SwiGLU; derin bir yoğun ağda normalizasyon istemiyorsan SELU (koşullarıyla). Zaman kısıtı yoksa doğrulama kümesiyle birkaçını dene.

### Hook'larla teşhis {#hook}

Eğitim kötü gidiyorsa ilk bakılacak yer katman başına aktivasyon ve gradyan istatistikleridir:

```python
stats = {}
def fwd_hook(name):
    def h(mod, inp, out): stats[name] = (out.mean().item(), out.std().item(), (out <= 0).float().mean().item())
    return h
for name, m in model.named_modules():
    if isinstance(m, nn.Linear): m.register_forward_hook(fwd_hook(name))
loss.backward()
for name, p in model.named_parameters():
    print(name, p.grad.norm().item() / (p.norm().item() + 1e-12))   # güncelleme/ağırlık oranı
```

Sağlıklı bir ağda aktivasyon std'leri katmanlar boyunca kabaca sabit, ölü ReLU oranı makul (<%50), gradyan normları katmanlar arasında birkaç büyüklük mertebesinden fazla değişmez; $\eta\cdot\|\nabla W\|/\|W\|\approx10^{-3}$ civarında bir güncelleme oranı iyi bir işarettir (Karpathy). Modülün **d** aşamasında hook'larla bu istatistikleri toplayacaksın.

### Özet {#ozet}

| kavram | akılda kalacak |
|---|---|
| varyans çarpanı | $\mathrm{Var}(z)=n_{\text{in}}\mathrm{Var}(w)\mathbb E[x^2]$; katman başına çarpan 1 olmalı |
| Glorot | $\mathrm{Var}(w)=2/(n_{\text{in}}+n_{\text{out}})$: tanh, sigmoid, doğrusal |
| He | $\mathrm{Var}(w)=2/n_{\text{in}}$: ReLU ailesi (2, ReLU'nun ½'sini telafi eder) |
| LeCun | $\mathrm{Var}(w)=1/n_{\text{in}}$: SELU |
| sigmoid | sıfır merkezli değil, $\sigma'\le1/4$, doyar |
| ReLU | doymaz, ucuz; ölü nöron riski → Leaky/ELU/GELU |
| teşhis | hook'larla katman başına ortalama/std/ölü oran ve gradyan normları |

**Kaynaklar:** Toygar, BİL 442 Bölüm 11 slaytları (başlatma tablosu, sigmoid'in sorunları, ReLU ailesi, SELU koşulları, "hangisini kullanmalı") ve Alıştırma 2; Géron, *Hands-On ML* (3. bs.) Bölüm 11; Glorot ve Bengio (2010); He ve ark. (2015), "Delving deep into rectifiers" (He başlatma, PReLU); Klambauer ve ark. (2017), SELU; Clevert ve ark. (2016), ELU; Xu ve ark. (2015), ReLU ailesinin karşılaştırması (RReLU); Hendrycks ve Gimpel (2016), GELU; Ramachandran ve ark. (2017), Swish; Karpathy (2019), "A recipe for training neural networks".
