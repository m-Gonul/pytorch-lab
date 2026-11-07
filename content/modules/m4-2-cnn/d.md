- `BasicBlock(in_c, out_c, stride=1)` — `nn.Module`; conv'lar `bias=False`, projeksiyon gerekiyorsa `self.shortcut = nn.Sequential(conv1×1, BN)`, gerekmiyorsa `nn.Identity()`. `forward`: `relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))`.
- `SmallResNet(num_classes)` — stem: `conv3×3 (3→16) + BN + ReLU`; üç aşama, her biri 2 BasicBlock: 16 (stride 1), 32 (stride 2), 64 (stride 2); `AdaptiveAvgPool2d(1)` + `Linear(64, num_classes)`.

Kontrol parametre sayısını, shape'leri, shortcut türlerini ve küçük bir sentetik görevde öğrenmeyi denetler.
