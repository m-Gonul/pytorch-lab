Bir tensör, düz bir bellek bloğu (**storage**) üzerine konmuş bir bakış açısıdır. Bakışı tanımlayan üç şey:

- **shape** — her eksenin uzunluğu,
- [[stride]] — o eksende bir adım ilerlemek için bellekte kaç **eleman** atlanacağı,
- **offset** — ilk elemanın storage içindeki konumu.

$$
\text{adres}(i_0,\dots,i_{k-1})=\text{offset}+\sum_{j}i_j\cdot\text{stride}_j .
$$

Row-major (C sırası) bir $(n, m)$ tensörün stride'ı $(m,1)$'dir. Transpoze, dilim (`x[:, ::2]`), `expand`, `unfold` ve `view` yeni bellek ayırmaz; yalnız shape/stride/offset üçlüsünü değiştirir. Bunlar O(1) işlemlerdir. Buna karşılık [[contiguous]] olmayan bir tensör üzerinde çalışan bazı kernel'lar önce kopya ister; gizli kopyalar ileri PyTorch'ta en yaygın performans kaybıdır.

Modülün ikinci yarısı ölçüm: ısınma (ilk çağrıda lazy init, önbellek, `torch.compile`), birden çok tekrar, ortalama değil **medyan**, GPU'da senkronizasyon.
