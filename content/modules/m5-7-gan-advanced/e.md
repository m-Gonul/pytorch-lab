StyleGAN yapı taşları.

- `MappingNet(z=64, w=64, L=8)` — piksel normu ($z/\sqrt{\mathrm{mean}(z^2)+\epsilon}$) + L × (doğrusal + LeakyReLU 0.2).
- `AdaIN(C, w)` — w'den doğrusal izdüşümle `(y_s, y_b)` (`y_s` başlangıçta 1 civarı: bias 1), örnek ve kanal başına normalizasyon.
- `NoiseInjection(C)` — kanal başına öğrenilen ölçekle `x + s_c · noise`.
- `modulated_conv(x, weight, style, demod=True)` — StyleGAN2: $w'_{ijk}=s_i\,w_{ijk}$, $w''=w'/\sqrt{\sum_{i,k}w'^2+\epsilon}$; örnek başına ağırlık → grup konvolüsyonu (batch'i gruplara katlayarak).
- `truncate(w, w_avg, psi)`.
