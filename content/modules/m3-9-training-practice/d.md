PyTorch, sklearn `load_digits` (8×8 = 64 öznitelik; Fashion-MNIST'in küçük vekili).

- `capacity_runs(epochs=40, seed=0)` → `{mimari: [(eğitim kaybı, doğrulama kaybı, doğrulama doğruluğu) epoch başına]}` — mimariler ders alıştırmasının ölçeklenmiş hâli: 64-300-100-10, 64-400-500-300-100-10, 64-600-700-500-400-300-100-10; Adam 1e-3, batch 32, eğitim 1000 / doğrulama 797.
- `transfer(n_target=100, seed=0)` → `{"sıfırdan", "dondurulmuş", "ince ayar", "ilgisiz kaynak"}` doğrulukları. **Yardımcı görevle ön-eğitim:** gizli 4 boyutlu etkenler $z$, gözlem $x=\sin(2\tanh(2zA_1)A_2)+\text{gürültü}\in\mathbb R^{64}$ (sabit rastgele $A_1, A_2$). Kaynak (bol etiket, 5000): $z$'yi regresyonla tahmin et (gövde 64-128-32). Hedef (az etiket, `n_target`): $\mathbb 1[z_3+z_4>0]$. (a) sıfırdan, (b) gövde dondurulmuş + yeni başlık, (c) önce başlık sonra tümü 1e−4 ile, (d) gövde **rastgele etiketlerle** ön-eğitilmiş (dondurulmuş).

Not: digits'te 0–4'ten 5–9'a aktarım burada **zarar** verir (piksel uzayı zaten neredeyse doğrusal ayrılabilir, kaynak öznitelikleri 5 sınıfa özelleşir) — negatif aktarım da bir derstir.
