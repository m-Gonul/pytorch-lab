ZeRO-1'de parametre (4) ve gradyan (4) her GPU'da tam; yalnız Adam'ın iki momenti (8 bayt) 8'e bölünüyor: $4+4+1=9$ bayt/parametre. ZeRO-3'te her şey bölünür: $16/8=2$.
