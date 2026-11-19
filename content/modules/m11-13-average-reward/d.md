**Yarı gradyanı autograd ile kur.** `Q`44×2 boyutlu, `float64` yaprak tensor olsun: durum indisi `4*free+priority`. Bu, bir-sıcak özellikli doğrusal modelin tabloya denk özel durumudur. `differential_torch_step` yalnız `Q[state,action]` üzerinden yarım kare kaybı türevlesin. Sonraki Q **detach** edilmelidir; ortalama ödül kestirimi de bir optimizasyon parametresi değildir.

Fonksiyon yeni yaprak Q ve yeni rho döndürsün. Aynı TD hatasının sayısal değeri rho güncellemesinde kullanılır. `torch_sarsa` bu adımı kaynak ortamında5000 gerçek geçişte uygulasın; `Q,rho,reward_mean,visits` döndürsün. Yeni eylem güncellemeden önce seçilir. `visits`44elemanlıdır. Yeni kabul edilen işi de bırakma çekilişine kat; p=.06. Bu aşama model-oracle'ına bakmaz.

Hedefi türevlemek yalnız bir API hatası değildir: sonraki durum parametrelerini de değiştirir ve kaynak yarı-gradyan kuralını başka bir algoritmaya çevirir. Kontrol, basit bir geçişin doğru gradyan desteğini ayrıca denetler.
