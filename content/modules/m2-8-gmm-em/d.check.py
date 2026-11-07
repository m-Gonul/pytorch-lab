import math
import torch
from labcheck import close, need, ensure


def em_ref(X, mus, iters=60):
    n, d = X.shape; K = mus.shape[0]
    pis = torch.full((K,), 1.0 / K, dtype=X.dtype); covs = torch.eye(d, dtype=X.dtype).repeat(K, 1, 1)
    for _ in range(iters):
        diff = X[:, None] - mus[None]
        maha = torch.einsum("nkd,kde,nke->nk", diff, torch.linalg.inv(covs), diff)
        L = pis.log() - 0.5 * (d * math.log(2 * math.pi) + torch.logdet(covs) + maha)
        Rr = (L - torch.logsumexp(L, 1, keepdim=True)).exp()
        Nk = Rr.sum(0); pis = Nk / n; mus = Rr.T @ X / Nk[:, None]
        diff = X[:, None] - mus[None]
        covs = torch.einsum("nk,nkd,nke->kde", Rr, diff, diff) / Nk[:, None, None] + 1e-6 * torch.eye(d, dtype=X.dtype)
    diff = X[:, None] - mus[None]
    L = pis.log() - 0.5 * (d * math.log(2 * math.pi) + torch.logdet(covs) + torch.einsum("nkd,kde,nke->nk", diff, torch.linalg.inv(covs), diff))
    return float(torch.logsumexp(L, 1).mean())


def check(ns, R):
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(150, 2, generator=g, dtype=torch.float64) @ torch.tensor([[1.0, 0.8], [0, 0.6]], dtype=torch.float64),
                   torch.randn(100, 2, generator=g, dtype=torch.float64) * 0.5 + torch.tensor([4.0, 0.0], dtype=torch.float64)])

    def st():
        raw = torch.randn(3, 2, 2, generator=g, dtype=torch.float64)
        L = need(ns, "make_scale_tril")(raw)
        ensure(torch.equal(L, torch.tril(L)), "alt üçgen değil")
        ensure(bool((torch.diagonal(L, dim1=-2, dim2=-1) > 0).all()), "köşegen pozitif değil")
    R.test("make_scale_tril: alt üçgen, pozitif köşegen", st)

    def ll():
        logits = torch.tensor([0.3, -0.2], dtype=torch.float64)
        mus = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
        raw = torch.randn(2, 2, 2, generator=g, dtype=torch.float64) * 0.3
        Ls = torch.tril(raw, -1) + torch.diag_embed(torch.diagonal(raw, dim1=-2, dim2=-1).exp())
        covs = Ls @ Ls.transpose(-1, -2)
        diff = X[:, None] - mus[None]
        Lg = torch.log_softmax(logits, 0) - 0.5 * (2 * math.log(2 * math.pi) + torch.logdet(covs) + torch.einsum("nkd,kde,nke->nk", diff, torch.linalg.inv(covs), diff))
        close(need(ns, "gmm_mean_ll")(X, logits, mus, raw), torch.logsumexp(Lg, 1).mean())
    R.test("MixtureSameFamily = elle formül", ll)

    def fit():
        mus0 = X[[0, 200]].clone()
        v = need(ns, "fit_gmm_gd")(X, mus0)
        ref = em_ref(X, mus0)
        ensure(isinstance(v, float), "float döndürmeli")
        ensure(v > ref - 0.02, f"gradyan ile ortalama ll {v:.4f}, EM {ref:.4f}")
        return f"ortalama log-likelihood: gradyan {v:.4f}, EM (60 iterasyon) {ref:.4f}"
    R.test("gradyanla eğitim EM'in çözümüne ulaşıyor", fit)
