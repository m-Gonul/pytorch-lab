CPU'da `torch.autocast("cpu", dtype=torch.bfloat16)`, GPU'da `("cuda", torch.float16)` + `GradScaler`.

- `amp_dtypes(model, x, y, device_type, dtype)` → `{"linear": …, "loss": …, "param": …, "grad": …}`: autocast içinde `model(x)` çıktısının ve `F.cross_entropy` kaybının dtype'ı, geri geçişten sonra ilk parametrenin ve gradyanının dtype'ı.
- `train_step_amp(model, opt, scaler, x, y, device_type, dtype)` → kayıp (float): autocast'li ileri geçiş, `scaler.scale(loss).backward()`, `scaler.step(opt)`, `scaler.update()`, `opt.zero_grad()`.
