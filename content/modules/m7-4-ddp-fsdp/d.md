İki CPU süreci, `gloo`. Süreçler `torch.multiprocessing.start_processes(…, start_method="fork")` ile başlatılır (Windows'ta `fork` yok: WSL ya da Colab kullan).

- `allreduce_mean(t)` — `dist.all_reduce(t, op=SUM)`, sonra `world_size`'a böl (yerinde).
- `worker(rank, world, port, out, steps, manual)` — `init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", …)`; `torch.manual_seed(0)` ile `make_model()`; veri `make_data()`, işçi $r$ satırları `r::world` alır; SGD (`lr=0.1`), MSE.
  - `manual=False`: modeli `DistributedDataParallel` ile sar.
  - `manual=True`: DDP yok; `backward`'dan sonra her gradyana `allreduce_mean`.
  - Sonunda parametreleri `out/r{rank}.pt`'ye kaydet, `destroy_process_group()`.
- `run(world, steps, manual)` — boş bir port bul, süreçleri başlat, kaydedilen parametre listelerini döndür.

Dikkat: eşdeğerlik için işçi parçaları **eşit** boyutta olmalı. 64 örneği 3 işçiye bölersen (22, 21, 21) parça ortalamalarının ortalaması tam batch ortalamasına eşit olmaz; `DistributedSampler` bu yüzden veriyi doldurur ya da keser.
