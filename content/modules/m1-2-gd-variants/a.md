Önce tek boyutlu kuadratikte düz GD ile momentumu yan yana koy. Düz GD minimuma her adımda kalan mesafenin yarısı kadar yaklaşıyor ($w=0\to1.5\to2.25\to\dots$). Momentum ikinci adımda biriktirdiği hızla tam $3$'e ulaşıyor. Bu şanslı bir seçim; daha büyük $\beta$ ile hedefi aşar ve salınır.

Sonra regresyon verisinde tek örnekli ve iki örnekli gradyanları hesapla. M1.1'deki tam batch gradyanı $\partial_w=-16.67$ idi; SGD ($-4$) ve mini-batch ($-23$) aynı **yönü** gösteriyor ama büyüklükleri farklı. Ortalamaları alınırsa tam gradyana yaklaşırlar.
