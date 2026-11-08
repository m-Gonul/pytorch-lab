<!-- çözüm -->

Batch büyütmek verimi artırır ama tek isteğin gecikmesini de artırır (5 ms → 20 ms). Sunucular bu yüzden "dinamik batching" yapar: kısa bir süre bekleyip gelen istekleri birleştirir.

İzlenen graf yalnız $2x$ dalını içeriyor: negatif girdide de $2x$ hesaplanıyor.
