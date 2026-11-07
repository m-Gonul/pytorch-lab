**Nesnel kalite** ölçütleri, insanların öznel değerlendirmesini (pahalı, yavaş) taklit etmeye çalışır. Referansın ne kadarının bilindiğine göre: **tam referanslı** (MSE, PSNR, SSIM), **indirgenmiş referanslı** (yalnız bazı öznitelikler), **referanssız** (yalnız bozuk görüntü).

**MSE / PSNR:** $\text{MSE}=\frac1N\sum(x_i-y_i)^2$, $\text{PSNR}=10\log_{10}\frac{R^2}{\text{MSE}}$. Basit, türevlenebilir, fiziksel anlamı açık; ama hata türüne kör: ortalama kayması, kontrast germe, bulanıklık ve JPEG aynı MSE'de çok farklı görünür.

**SSIM:** doğal görüntüler yapılıdır (komşu pikseller güçlü bağımlı) ve görme sistemi yapıyı çıkarmaya uyarlanmıştır. Benzerliği üçe ayır:

- parlaklık $l=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}$,
- kontrast $c=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}$,
- yapı $s=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}$,

$C_1=(K_1L)^2$, $C_2=(K_2L)^2$, $K_1=0.01$, $K_2=0.03$. $\alpha=\beta=\gamma=1$, $C_3=C_2/2$ ile

$$\text{SSIM}=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.$$

Özellikler: simetri, $\le1$, ve yalnız $x=y$ iken 1. İstatistikler **yerel pencerede** (11 × 11 Gauss, $\sigma=1.5$) hesaplanır; SSIM haritasının ortalaması küresel SSIM'dir.

**MS-SSIM:** görüntüyü 5 ölçekte (her seferinde 2 × 2 ortalama + alt örnekleme) karşılaştır; kontrast–yapı terimleri her ölçekte, parlaklık yalnız en kabada: $\prod_jcs_j^{\beta_j}\cdot\text{SSIM}_M^{\beta_M}$.
