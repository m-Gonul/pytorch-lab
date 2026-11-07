Bu modül bir mini proje: [[minGPT]]'nin "adder" örneğinden esinlenen, iki basamaklı toplamayı öğrenen bir GPT.

**Veri:** her örnek bir satır, `ab+cd=xyz⏎`; cevap 3 basamağa sıfırla doldurulup **ters çevrilir** (52 → `250`). Sözlük: `0123456789+=⏎` (13 token). Model girdi olarak ilk 9 token'ı görür, her konumda sonrakini tahmin eder; **kayıp yalnız cevap basamaklarını** tahmin eden 3 konumda.

**Eğitim döngüsü** (önceki modüllerden):

- AdamW (M3.4), öğrenme oranı: doğrusal warmup, sonra cosine düşüş:

$$
\eta_t=\begin{cases}\eta_{\max}\frac{t+1}{W} & t<W\\[2pt] \eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\big(1+\cos\pi\frac{t-W}{T-W}\big) & t\ge W\end{cases}
$$

- gradyan normu kırpma (`clip_grad_norm_`), değerlendirmede greedy çözme (M4.7).

**Bütçe:** yoğun bir Transformer'ın eğitimi $\approx6ND$ FLOP ($N$ parametre, $D$ token: ileri $2ND$, geri $4ND$). Chinchilla'ya göre hesap-optimal $D\approx20N$.

**Profil:** `torch.profiler.profile` hangi işlemin ne kadar sürdüğünü gösterir; çok sayıda küçük işlem (örneğin head başına ayrı matmul) Python ve kernel başlatma yükü demektir.
