Triton (GPU yoksa yorumlayıcı).

- `scan(x, BLOCK=1024)` — float32 kapsayıcı tarama, hiyerarşik: blok çekirdeği `tl.cumsum` + blok toplamları; toplamlar özyinelemeli taranır; ofset çekirdeği her bloğa önceki toplamı ekler.
- `linear_recurrence(a, b)` — `a, b (B, L)`, L ≤ 1024: her program bir satır; `tl.associative_scan((a, b), 0, combine)`; `combine` afin bileşke.
