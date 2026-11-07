import warnings
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    warnings.filterwarnings("ignore")
    Z = data.astronaut()[::4, ::4].reshape(-1, 3).astype(np.float64) / 255
    rng = np.random.default_rng(0)
    mu0 = Z[rng.choice(len(Z), 5, replace=False)]

    def sk():
        mu, cov, w, ll = need(ns, "gmm_em")(torch.tensor(Z), torch.tensor(mu0), 30)
        c0 = np.cov(Z.T) + 1e-6 * np.eye(3)
        g = GaussianMixture(5, covariance_type="full", weights_init=np.full(5, 0.2), means_init=mu0,
                            precisions_init=np.stack([np.linalg.inv(c0)] * 5), max_iter=30, tol=0, reg_covar=1e-6).fit(Z)
        close(mu.numpy(), g.means_, atol=1e-8, what="ortalamalar")
        close(cov.numpy(), g.covariances_, atol=1e-8, what="kovaryanslar")
        close(w.numpy(), g.weights_, atol=1e-8, what="ağırlıklar")
        close(float(ll), g.score(Z), atol=1e-8, what="ortalama log-olabilirlik")
        return f"30 iterasyon, log-olabilirlik {float(ll):.4f}"
    R.test("gmm_em = sklearn GaussianMixture", sk)

    def ell():
        g = np.random.default_rng(3)
        A = np.array([[3.0, 0], [0, 0.3]])
        rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        X = np.concatenate([g.normal(size=(300, 2)) @ (rot(0.6) @ A).T, g.normal(size=(300, 2)) @ (rot(0.6) @ A).T + [0, 2.2]])
        y = np.repeat([0, 1], 300)
        init = X[[0, 300]]
        km = KMeans(2, init=init, n_init=1).fit(X).labels_
        mu, cov, w, _ = ns["gmm_em"](torch.tensor(X), torch.tensor(init), 100)
        from scipy.stats import multivariate_normal as mvn
        lab = np.argmax(np.stack([w[k].item() * mvn(mu[k].numpy(), cov[k].numpy()).pdf(X) for k in range(2)], 1), 1)
        a_km, a_g = adjusted_rand_score(y, km), adjusted_rand_score(y, lab)
        ensure(a_g > a_km + 0.2, f"GMM ARI {a_g:.2f}, k-means ARI {a_km:.2f}")
        R.table("İki paralel, uzamış küme", ["yöntem", "ARI"], [["k-means (küresel)", f"{a_km:.2f}"], ["GMM (tam kovaryans)", f"{a_g:.2f}"]],
                note="k-means kümeleri ortalamaya Öklid uzaklığıyla ayırır; eliptik ve yakın kümelerde sınırı yanlış yere çizer.")
    R.test("eliptik kümelerde GMM > k-means", ell)
