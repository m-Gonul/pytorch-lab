import copy
import time
import torch
from labcheck import need, ensure


def check(ns, R_):
    def t_mask():
        g = torch.Generator().manual_seed(0)
        x, _ = need(ns, "corpus")(2000, 5)
        inp, lab = need(ns, "mlm_mask")(x, g, 64)
        sel = lab != -100
        ensure(not sel[:, 0].any() and abs(sel.float().mean().item() - 0.15 * 16 / 17) < 0.01, "seçim")
        ensure(abs((inp[sel] == 3).float().mean().item() - 0.8) < 0.03 and torch.equal(inp[~sel], x[~sel]), "80/10/10")
    R_.test("MLM maskesi", t_mask)

    def t_train():
        U, _ = ns["corpus"](20000, 0)
        Xl, yl = ns["corpus"](50, 1)
        Xt, yt = ns["corpus"](2000, 2)
        torch.manual_seed(0)
        pre = need(ns, "TinyBERT")(64)
        t0 = time.perf_counter()
        L = need(ns, "pretrain")(pre, U, 4, 3e-3, 0)
        tp = time.perf_counter() - t0
        rows, acc = [], {}
        for name, mk, fr in (("sıfırdan (50 etiket)", lambda: (torch.manual_seed(1), ns["TinyBERT"](64))[1], False),
                             ("MLM ön eğitim + ince ayar", lambda: copy.deepcopy(pre), False),
                             ("MLM ön eğitim, dondurulmuş + doğrusal sonda", lambda: copy.deepcopy(pre), True)):
            m = need(ns, "finetune")(mk(), Xl, yl, 100, 1e-3, fr)
            m.eval()
            with torch.no_grad():
                acc[name] = (m.cls(m(Xt)[:, 0]).argmax(1) == yt).float().mean().item()
            rows.append([name, f"{acc[name]:.3f}"])
        a = list(acc.values())
        ensure(a[1] > a[0] + 0.04 and a[2] > a[0], f"ön eğitim yardımcı olur {acc}")
        R_.table(f"Konu sınıflandırma, 50 etiketli örnek (ön eğitim: 20 000 etiketsiz cümle, 4 epoch, {tp:.0f} sn, son MLM kaybı {L:.2f})", ["yöntem", "test doğruluğu"], rows,
                 note="Etiketsiz metinde maskeli kelimeleri tahmin etmek için model hangi içerik kelimelerinin birlikte geçtiğini (konuyu) ve işlev kelimelerinin içerik kelimelerine bağımlılığını öğrenmek zorunda; bu temsiller az etiketle aşağı akış görevine aktarılır. Dondurulmuş temsillere yalnız doğrusal bir katman eklemek bile sıfırdan eğitmeyi geçer (ELMo/öznitelik tabanlı kullanım); tüm ağırlıkları ince ayarlamak (GPT-1, BERT) daha da iyi.")
    R_.test("ön eğitim + ince ayar", t_train)
