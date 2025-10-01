#!/usr/bin/env python3
import botorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from torch.distributions import Categorical, Laplace, Independent, MixtureSameFamily
from torch.func import vmap, jacrev, jacfwd

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
torch.set_default_dtype(dtype)

width, height = 1.2, 0.8

plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.labelcolor": "black",
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 7,
    "text.usetex": True,
    "pgf.rcfonts": True,
    "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{amsmath}\usepackage{bm}\usepackage{mathtools}\usepackage{dsfont}\usepackage{amssymb}\usepackage{pifont}\newcommand{\cmark}{\ding{51}}\newcommand{\xmark}{\ding{55}}",
})

means = torch.tensor([[0, 1], [0, 1]], device=device, dtype=dtype)
b_scales = torch.tensor([[4, 2], [4, 2]], device=device, dtype=dtype)
weights = torch.tensor([0.30, 0.70], device=device, dtype=dtype)

cat = Categorical(weights)
comp = Independent(Laplace(loc=means, scale=b_scales), reinterpreted_batch_ndims=1)
pX = MixtureSameFamily(cat, comp)


def _log_normlaplace_1d(y, mu, b, sigma):
    z = y - mu
    a1 = -(z / sigma) - (sigma / b)
    a2 = (z / sigma) - (sigma / b)
    t1 = (z / b) + botorch.utils.probability.utils.log_ndtr(a1)
    t2 = (-z / b) + botorch.utils.probability.utils.log_ndtr(a2)
    return (sigma ** 2) / (2 * b ** 2) - torch.log(2 * b) + torch.logaddexp(t1, t2)


def log_pY_given_gamma(y, gamma):
    if y.ndim == 1:
        y = y.unsqueeze(0)
    sigma = torch.sqrt(gamma)
    lp0 = _log_normlaplace_1d(y[:, 0].unsqueeze(1), means[:, 0], b_scales[:, 0], sigma)
    lp1 = _log_normlaplace_1d(y[:, 1].unsqueeze(1), means[:, 1], b_scales[:, 1], sigma)
    comp_lp = lp0 + lp1
    log_w = torch.log(weights / weights.sum())
    return torch.logsumexp(comp_lp + log_w, dim=-1) - y.shape[-1] * torch.log(gamma)


def hessian_y_logpY_gamma():
    return jacfwd(jacrev(lambda y, g: log_pY_given_gamma(y, g)), argnums=0)


def compute_H_per_x(X, gammas, num_noises=1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    X, gammas = X.to(device), gammas.to(device)
    dgam = gammas[1:] - gammas[:-1]
    bs, num_gamma = X.shape[0], gammas.shape[0]
    eps = torch.randn((num_noises, bs, num_gamma, 2), device=device, dtype=dtype)
    ggrid = gammas.view(1, 1, num_gamma, 1)
    y = ggrid * X.view(1, bs, 1, 2) + ggrid.sqrt() * eps
    y_flat, g_flat = y.reshape(-1, 2), ggrid.expand(num_noises, bs, num_gamma, 1).reshape(-1)
    H = vmap(hessian_y_logpY_gamma())(y_flat, g_flat).view(num_noises * bs * num_gamma, 2, 2)
    H = torch.bmm(H, H).view(num_noises, bs, num_gamma, 2, 2).mean(0)
    return ((H * (gammas ** 2).view(1, num_gamma, 1, 1))[:, :-1] * dgam.view(1, num_gamma - 1, 1, 1)).sum(1)


# --- Plotting ---
def contour(f, xlim, ylim, n):
    xs, ys = np.linspace(*xlim, n), np.linspace(*ylim, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([XX, YY], -1), device=device, dtype=dtype).reshape(-1, 2)
    with torch.no_grad():
        Z = f(grid).reshape(n, n).cpu().numpy()
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("none")
    levels = np.linspace(Z.min(), Z.max(), 20)
    ax.contourf(XX, YY, Z, levels=levels, cmap=plt.cm.viridis, antialiased=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_zorder(10000)
    plt.tight_layout()
    plt.subplots_adjust(left=0.0, right=1, bottom=0.0, top=1)
    return fig, ax


def plot_logpX_with_ellipses(X_centers, H_per_x, xlim=(-5, 5), ylim=(-5, 5),
                             ellipse_scale=1.0, path="hessian_ellipses.pdf"):
    _, ax = contour(lambda z: pX.log_prob(z), xlim, ylim, n=300)
    Xc, Hc = X_centers.cpu().numpy(), H_per_x.cpu().numpy()
    for (cx, cy), H in zip(Xc, Hc):
        lam, V = np.linalg.eigh(H)
        r = ellipse_scale / np.sqrt(lam)
        i = int(np.argmax(r))
        j = 1 - i
        ang = np.degrees(np.arctan2(V[1, i], V[0, i]))
        ax.add_patch(Ellipse((cx, cy), width=2 * r[i], height=2 * r[j], angle=ang,
                             fill=False, edgecolor="k", linewidth=0.3))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    grid_size = 13
    lim = 4
    xlim = (-lim * width, lim * width)
    ylim = (-lim * height, lim * height)

    gx = torch.tensor(
        [-lim, -lim + 1.5, -lim + 2.3, -lim + 3, -lim + 3.5, 0, lim - 3.5, lim - 3, lim - 2.3, lim - 1.5, lim],
        device=device, dtype=dtype)
    gy = torch.tensor(
        [-lim, -lim + 1, -lim + 1.6, -lim + 2.1, -lim + 2.5, -lim + 3, -lim + 3.5, 0, lim - 3.5, lim - 3, lim - 2.5,
         lim - 2], device=device, dtype=dtype)

    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    X_centers = torch.stack([GX, GY], -1).reshape(-1, 2) + means[0].unsqueeze(0)

    gammas = torch.logspace(-2, 2, 200, device=device, dtype=dtype, base=2)
    H_per_x = compute_H_per_x(X_centers, gammas, num_noises=100, seed=123)

    plot_logpX_with_ellipses(X_centers, H_per_x, xlim=xlim, ylim=ylim,
                             ellipse_scale=0.02, path="hessian_ellipses.pdf")

