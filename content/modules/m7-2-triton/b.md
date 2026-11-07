Saf Python ile Triton'un blok modeli. "Bellek" düz bir liste, "işaretçi" bir indeks.

- `arange(n)` → `list(range(n))`.
- `load(mem, offs, mask, other=0.0)` → liste; maskeli konumlarda `other`.
- `store(mem, offs, vals, mask)` — yalnız maskeli konumlara yaz.
- `launch(program, grid, *args)` — her `pid in range(grid)` için `program(pid, *args)`.
- `add_program(pid, x, y, out, n, BLOCK)`.
- `softmax_program(pid, X, Y, n_cols, BLOCK)` — `X`, `Y` satır-ana düz liste; satır `pid`; maskelenenlere `other=-inf`.
