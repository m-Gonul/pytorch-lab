Saf Python.

- `amdahl(f, n)`, `gustafson(f, n)`, `karp_flatt(S, p)`.
- `pipeline_makespan(stages, n_items)` — her öğe aşamaları sırayla geçer, her aşama aynı anda tek öğe işler: $\text{bitiş}[i][s]=\max(\text{bitiş}[i][s-1],\text{bitiş}[i-1][s])+t_s$; son öğenin bitişi.
- `work_span(dur, preds)` — `dur` görev → süre sözlüğü, `preds` görev → öncül listesi; `(W, S∞)`.
- `list_schedule(dur, preds, p)` — olay güdümlü açgözlü zamanlama: hazır görevler arasından **alt seviyesi** (görevden çıkışa en uzun yol, kendi süresi dahil) en büyük olanı boş işlemciye ata; makespan döndür.
