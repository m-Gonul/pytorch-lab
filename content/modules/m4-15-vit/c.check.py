import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("patchify", "unpatchify", "patch_embed_conv", "conv_to_linear", "mean_attention_distance")])
    rng = np.random.default_rng(0)

    def t():
        x = rng.normal(size=(3, 8, 12))
        z = need(ns, "patchify")(x, 4)
        ensure(z.shape == (6, 48), "şekil")
        close(z[4], x[:, 4:8, 4:8].transpose(1, 2, 0).ravel(), what="yama 4 = satır 1, sütun 1; (p1, p2, c) sırası")
        close(need(ns, "unpatchify")(z, 4, 3, 8, 12), x, what="ters")
        Wc, b = rng.normal(size=(5, 3, 4, 4)), rng.normal(size=5)
        e1 = need(ns, "patch_embed_conv")(x, Wc, b, 4)
        E = need(ns, "conv_to_linear")(Wc)
        close(e1, z @ E + b, atol=1e-10, what="adım P'li konv. = patchify @ E")
        g = 14
        I = np.eye(g * g)
        close(need(ns, "mean_attention_distance")(I, g, 16), 0.0, what="kendine bakan baş: 0")
        U = np.full((g * g, g * g), 1 / (g * g))
        local = np.zeros((g * g, g * g))
        yy, xx = np.divmod(np.arange(g * g), g)
        nb = (np.abs(yy[:, None] - yy[None]) <= 1) & (np.abs(xx[:, None] - xx[None]) <= 1)
        local[nb] = 1
        local /= local.sum(1, keepdims=True)
        rows = [["yalnız kendisi", f"{ns['mean_attention_distance'](I, g, 16):.1f}"], ["3 × 3 komşuluk (CNN benzeri)", f"{ns['mean_attention_distance'](local, g, 16):.1f}"], ["düzgün (küresel)", f"{ns['mean_attention_distance'](U, g, 16):.1f}"]]
        ensure(float(rows[1][1]) < float(rows[2][1]), "yerel < küresel")
        R_.table("Ortalama attention uzaklığı (224 × 224, P = 16, piksel)", ["attention deseni", "uzaklık"], rows,
                 note="Makale Şekil 7: ViT'in ilk katmanlarında bazı başlar zaten neredeyse tüm görüntüye bakar (~100+ piksel), bazıları yereldir (CNN'in ilk katmanlarındaki receptive field gibi); derinlikle tüm başlar küreselleşir. Bu ölçü, yerelliğin önsel olarak verilmediğini, veriden öğrenildiğini gösterir.")
    R_.test("yamalar ve attention uzaklığı", t)
