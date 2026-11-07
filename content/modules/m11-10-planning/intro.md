**Modelsiz** (Q-öğrenme, politika gradyanı, aktör–eleştirmen) ve yalnız ödül modelli (haydutlar) yöntemlerden sonra: **açık geçiş ve ödül modeli** öğren ve onunla **planla**. Kazanç: örnek verimliliği. Bedel: karmaşıklık, model hatası.

**Döngü:** eylem seç ve uygula → durum ve ödülü gözle → modeli güncelle → değeri/politikayı güncelle. Modeller: tablo (sayımlar: $\hat P(s'|s,a)=n(s,a,s')/n(s,a)$), doğrusal ($s'\approx As+Ba$), doğrusal olmayan (sinir ağı; rastgele ya da deterministik).

**Kısmi planlama:** karmaşık modelde her adımda değeri tam optimize etmek pahalı; bunun yerine birkaç simüle Q-öğrenme (ya da politika gradyanı) adımı → **Dyna-Q**. Tekrar tamponuna çok benzer: tampon basit ve gerçek örnekler, genelleme yok; model karmaşık, simüle örnekler, başka (s,a)'lara genelleyebilir.

**Şimdiki durumdan planla:** rastgele $\hat s$'lerde değil, şu anki durumdan ileri bak — bir sonraki eylemi iyileştirir. Sürekli kontrolde **MPC** (model öngörülü kontrol: birçok eylem dizisini modelde dene, en iyisinin ilk eylemini uygula, tekrarla). Ayrık karar problemlerinde **Monte Carlo ağaç araması** üç fikri birleştirir:

- yaprak düğümleri: değeri varsayılan politikayla (rastgele oyun) kestir;
- şans düğümleri: beklentiyi modelden örnekleyerek yaklaşıkla;
- karar düğümleri: yalnız umut veren eylemleri genişlet.

Dört adım: **seçim** (UCT: $\arg\max_a\bar Q(s,a)+c\sqrt{\ln N(s)/n(s,a)}$), **genişletme**, **simülasyon**, **geri yayılım**. AlphaGo'nun varyantında her kenar $Q(s,a)$, $\pi(a|s)$ (politika ağının önseli) ve $n(s,a)$ saklar; $a^*=\arg\max_aQ(s,a)+u(s,a)$, $u\propto\pi(a|s)/(1+n(s,a))$.
