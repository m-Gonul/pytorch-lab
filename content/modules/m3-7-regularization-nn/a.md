<!-- çözüm -->

Dropout'un çıktısını hesapla ve beklenen değeri kontrol et: $x_1=2$ yarı olasılıkla 4, yarı olasılıkla 0 oluyor; ortalaması 2. Ölçekleme olmasaydı ortalama 1 olurdu ve test zamanında her şeyi $1-p$ ile çarpmak gerekirdi (orijinal dropout makalesi böyle yapıyordu).

Early stopping'de kayıplar 2. epoch'tan sonra iki kez iyileşmiyor; 5. epoch'taki 0.61 asla görülmüyor. `patience` büyük seçilseydi görülürdü ama yine en iyi değil (0.6 < 0.61).
