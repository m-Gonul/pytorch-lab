`accum_step(model, opt, x, y, micro)` — `x`'i `micro` boyutlu parçalara böl; her parça için `loss = F.cross_entropy(…) * len(parça) / len(x)` ile `backward` (gradyanlar birikir), en sonda tek `opt.step()` ve `opt.zero_grad()`. Son parça daha küçük olabilir: ağırlık `len(parça) / len(x)` bu yüzden.

Kontrol, parametrelerin tek büyük batch adımıyla birebir aynı olduğunu ve mikro-batch küçüldükçe tepe aktivasyonun düştüğünü gösterir.
