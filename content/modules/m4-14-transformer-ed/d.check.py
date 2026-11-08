import time
import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def check(ns, R_):
    def t_layers():
        torch.manual_seed(0)
        d, h, ff = 16, 4, 32
        ref = torch.nn.TransformerEncoderLayer(d, h, ff, dropout=0.0, batch_first=True)
        mine = need(ns, "EncoderLayer")(d, h, ff, 0.0)
        mine.load_state_dict(ref.state_dict())
        x = torch.randn(3, 7, d, requires_grad=True)
        pad = torch.zeros(3, 7, dtype=torch.bool)
        pad[1, 5:] = True
        pad[2, 2:] = True
        a, b = ref(x, src_key_padding_mask=pad), mine(x, pad)
        close(a[~pad].detach().numpy(), b[~pad].detach().numpy(), atol=1e-5, what="EncoderLayer = nn.TransformerEncoderLayer")
        refd = torch.nn.TransformerDecoderLayer(d, h, ff, dropout=0.0, batch_first=True)
        md = need(ns, "DecoderLayer")(d, h, ff, 0.0)
        md.load_state_dict(refd.state_dict())
        y = torch.randn(3, 5, d, requires_grad=True)
        causal = torch.triu(torch.full((5, 5), float("-inf")), 1)
        close(refd(y, x, tgt_mask=causal, memory_key_padding_mask=pad).detach().numpy(), md(y, x, causal, pad).detach().numpy(), atol=1e-5, what="DecoderLayer = nn.TransformerDecoderLayer")
        y2 = y.detach().clone()
        y2[:, 3:] += 10
        ensure(torch.allclose(md(y2, x, causal, pad)[:, :3], md(y.detach(), x, causal, pad)[:, :3], atol=1e-5), "nedensel: gelecek token'lar geçmiş çıktıları etkilemez")
    R_.test("katmanlar PyTorch ile birebir", t_layers)

    def t_train():
        X, Y = need(ns, "dates")(12000, 0)
        Xt, Yt = ns["dates"](1000, 1)
        x, y = ns["encode"](X, Y)
        xt, yt = ns["encode"](Xt, Yt)
        yin = torch.cat([torch.full((len(y), 1), ns["SOS"]), y[:, :-1]], 1)
        rows, acc = [], {}
        for name, std in (("embedding N(0, 1/d) (doğru)", None), ("embedding N(0, 1) × √d", 1.0)):
            torch.manual_seed(0)
            m = need(ns, "Transformer")(64, 4, 128, 2, 0.1, std)
            ensure(tuple(m(x[:3], yin[:3]).shape) == (3, 11, ns["V_OUT"]), "logit şekli")
            opt = torch.optim.Adam(m.parameters(), 1.0, betas=(0.9, 0.98), eps=1e-9)
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 2 * 64 ** -0.5 * min((s + 1) ** -0.5, (s + 1) * 400 ** -1.5))
            t0 = time.perf_counter()
            for ep in range(2):
                m.train()
                perm = torch.randperm(len(x))
                for i in range(0, len(x), 64):
                    idx = perm[i:i + 64]
                    loss = F.cross_entropy(m(x[idx], yin[idx]).flatten(0, 1), y[idx].flatten(), label_smoothing=0.1)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sch.step()
            m.eval()
            p = m.greedy(xt)
            acc[name] = (p == yt).all(1).float().mean().item()
            with torch.no_grad():
                e = (m.src.weight * 8).norm(dim=1).mean().item()
            rows.append([name, f"{e:.1f}", f"{(p == yt).float().mean().item():.3f}", f"{acc[name]:.3f}", f"{time.perf_counter() - t0:.0f} sn"])
        a, b = acc.values()
        ensure(a > 0.5 and a > b + 0.3, f"doğruluklar {acc}")  # CPU/donanım farkına dayanıklı eşik
        R_.table("Tarih çevirisi: Transformer d = 64, N = 2 + 2, Noam (w = 400), ε_ls = 0.1, 2 epoch", ["başlatma", "‖emb·√d‖ (eğitim sonu)", "token doğr.", "tam eşleşme", "süre"], rows,
                 note="Sinüzoidal PE'nin her bileşeni [−1, 1]'de, vektör normu √(d/2) ≈ 5.7. N(0, 1) embedding × √d = 8 normu ~64 olur: konum sinyali gürültü düzeyinde kalır, model karakterlerin sırasını çözemez. Paylaşılan embedding'de (makale) başlatma ~N(0, 1/d) olduğu için √d çarpanı onu PE ile aynı ölçeğe getirir.")
    R_.test("eğitim", t_train)
