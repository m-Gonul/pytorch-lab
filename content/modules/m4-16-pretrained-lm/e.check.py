import itertools
import torch
import torch.nn.functional as F
from labcheck import close, need, ensure

V, T = 5, 4
ALL = torch.tensor(list(itertools.product(range(V), repeat=T)))


def reward(y):
    return (y == 0).sum(-1).float() - (y == 4).sum(-1).float()


class Policy(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.first = torch.nn.Parameter(torch.randn(V, generator=g) * 0.5)
        self.trans = torch.nn.Parameter(torch.randn(T - 1, V, V, generator=g) * 0.5)

    def logp(self, y):
        lp = F.log_softmax(self.first, -1)[y[:, 0]]
        for t in range(1, T):
            lp = lp + F.log_softmax(self.trans[t - 1], -1)[y[:, t - 1], y[:, t]]
        return lp


def check(ns, R_):
    def t():
        close(need(ns, "dpo_loss")(torch.tensor([-1.0]), torch.tensor([-5.0]), torch.tensor([-3.0]), torch.tensor([-4.0]), 0.1).item(), torch.log1p(torch.exp(torch.tensor(-0.3))).item(), atol=1e-6, what="DPO kaybı")
        ref = Policy(0)
        with torch.no_grad():
            lref = ref.logp(ALL)
        ensure(abs(lref.exp().sum().item() - 1) < 1e-5, "625 dizi = tam dağılım")
        r = reward(ALL)
        g = torch.Generator().manual_seed(1)
        idx = torch.multinomial(lref.exp(), 8000, True, generator=g).view(-1, 2)
        a, b = ALL[idx[:, 0]], ALL[idx[:, 1]]
        win = torch.rand(4000, generator=g) < torch.sigmoid(reward(a) - reward(b))
        yw, yl = torch.where(win[:, None], a, b), torch.where(win[:, None], b, a)
        with torch.no_grad():
            rw0, rl0 = ref.logp(yw), ref.logp(yl)
        rows = []
        res = {}
        for beta in (0.25, 0.5, 2.0):
            pol = Policy(0)
            opt = torch.optim.Adam(pol.parameters(), 0.05)
            for _ in range(300):
                loss = ns["dpo_loss"](pol.logp(yw), pol.logp(yl), rw0, rl0, beta)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                lp = pol.logp(ALL)
                star = need(ns, "optimal_logp")(lref, r, beta)
                res[beta] = (need(ns, "kl")(lp, star).item(), (lp.exp() * r).sum().item(), ns["kl"](lp, lref).item())
                rows.append([beta, f"{res[beta][1]:.3f}", f"{(star.exp() * r).sum().item():.3f}", f"{res[beta][2]:.3f}", f"{ns['kl'](star, lref).item():.3f}", f"{res[beta][0]:.3f}"])
        close(ns["kl"](lref, lref).item(), 0.0, atol=1e-6, what="KL(p‖p) = 0")
        ensure(res[0.5][0] < 0.1 and res[2.0][0] < 0.05, "DPO politikası analitik optimuma yakın")
        ensure(res[0.25][1] > res[0.5][1] > res[2.0][1] and res[0.25][2] > res[2.0][2], "β küçüldükçe ödül ve KL artar")
        R_.table(f"DPO, 4000 tercih çifti (referansın beklenen ödülü {(lref.exp() * r).sum().item():.3f})", ["β", "E_π[r]", "E_π*[r]", "KL(π‖π_ref)", "KL(π*‖π_ref)", "KL(π‖π*)"], rows,
                 note="Gerçek ödül Bradley–Terry ile tercih ürettiğinde DPO'nun optimumu KL cezalı RLHF'inkiyle aynıdır: π* ∝ π_ref · exp(r/β). Burada tüm diziler sayılabildiği için bunu doğrudan doğruluyoruz — ödül modeli, örnekleme ve PPO olmadan. β ödül–KL ödünleşimini ayarlar; küçük β'da sonlu veri ve optimizasyon hatası π'yi π*'dan uzaklaştırır.")
    R_.test("DPO = kapalı form optimum", t)
