- `fold_conv_bn(conv, bn)` → yeni `nn.Conv2d` (bias'lı): $\ s=\gamma/\sqrt{\sigma^2+\epsilon}$, $\ W'=W\cdot s$ (çıkış kanalı başına), $\ b'=(b-\mu)s+\beta$ (`conv.bias` yoksa $b=0$).
- `fuse_model(model)` → `torch.fx.GraphModule`: `symbolic_trace`; her `call_module` BN düğümünün girdisi bir Conv2d `call_module` düğümüyse ve conv çıktısının başka kullanıcısı yoksa, conv modülünü katlanmış olanla değiştir, BN düğümünü conv düğümüyle değiştir (`replace_all_uses_with`), BN düğümünü sil; `graph.lint()`, `recompile()`.

Kontrol modeli **eval modunda** karşılaştırır.
