Saf Python warp analizörleri.

- `sectors(byte_addrs, sector=32)` — bir warp'ın dokunduğu farklı sektör sayısı.
- `bank_degree(word_addrs, n_banks=32)` — çakışma derecesi: bank başına **farklı** adres sayısının en büyüğü (aynı adres yayın: 1 sayılır).
- `divergent_paths(conds)` — warp'taki thread'lerin dal koşullarından kaç farklı yol yürütüleceği.
- `occupancy(tpb, regs, smem, max_threads=2048, max_blocks=32, regs_sm=65536, smem_sm=102400)` → `(SM başına blok, doluluk oranı, sınırlayan etken)`; etken `"threads"`, `"regs"`, `"smem"` ya da `"blocks"`.
