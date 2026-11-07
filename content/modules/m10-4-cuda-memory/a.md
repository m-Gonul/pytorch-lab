Adres $4st$: $s=1$ → 0…124 B → 4 sektör; $s=2$ → 0…248 → 8; $s=32$ → her biri 128 B ara → 32 sektör (aktarılan 1024 B'nin yalnız 128'i kullanılır).

Bank $=$ kelime adresi mod 32: $2t$ → 0, 2, …, 30, 0, … (her banka 2 adres); $32t$ → hep bank 0; $33t$ → $t$ mod 32, hepsi farklı.

Doluluk: yazmaç sınırı $65536/(64\cdot256)=4$ blok, thread sınırı 8 → 4 blok × 256 = 1024 / 2048.
