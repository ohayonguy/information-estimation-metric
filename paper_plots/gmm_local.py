#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from torch.func import vmap, jacrev, jacfwd

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
torch.set_default_dtype(dtype)

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
    "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{amsmath}\usepackage{mathtools}\usepackage{dsfont}\usepackage{amssymb}\usepackage{pifont}\newcommand{\cmark}{\ding{51}}\newcommand{\xmark}{\ding{55}}",
})

width, height = 1.2, 0.8

means = torch.tensor([[0, 1],
                      [1, -1.]])

covs = torch.tensor([
    [[1, 0], [0, 0.1]],
    [[1, 0.5], [0.5, 0.4]],
])
weights = torch.tensor([0.3, 0.7], dtype=dtype, device=device)
log_w = weights.log()


def log_pX(x):
    mix = Categorical(logits=log_w, validate_args=False)
    comp = MultivariateNormal(loc=means, covariance_matrix=covs, validate_args=False)
    return MixtureSameFamily(mix, comp, validate_args=False).log_prob(x)


def log_pY_given_gamma(y, gamma):
    mean = gamma * means
    cov = (gamma ** 2) * covs + gamma * torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
    mix = Categorical(logits=log_w, validate_args=False)
    comp = MultivariateNormal(loc=mean, covariance_matrix=cov, validate_args=False)
    return MixtureSameFamily(mix, comp, validate_args=False).log_prob(y)


def hessian_y_logpY_gamma():
    return jacfwd(jacrev(lambda y, g: log_pY_given_gamma(y, g)), argnums=0)


def compute_H_per_x(X, gammas, num_noises=1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    X = X.to(device=device, dtype=dtype)
    gammas = gammas.to(device=device, dtype=dtype)
    dgam = gammas[1:] - gammas[:-1]
    bs, num_gamma = X.shape[0], gammas.shape[0]
    eps = torch.randn((num_noises, bs, num_gamma, 2), device=device, dtype=dtype)
    ggrid = gammas.view(1, 1, num_gamma, 1)
    y = ggrid * X.view(1, bs, 1, 2) + ggrid.sqrt() * eps
    y_flat = y.reshape(-1, 2)
    g_flat = ggrid.expand(num_noises, bs, num_gamma, 1).reshape(-1)
    H = vmap(hessian_y_logpY_gamma())(y_flat, g_flat).view(num_noises * bs * num_gamma, 2, 2)
    H = torch.bmm(H, H).view(num_noises, bs, num_gamma, 2, 2).mean(0)
    return ((H * (gammas ** 2).view(1, num_gamma, 1, 1))[:, :-1] * dgam.view(1, num_gamma - 1, 1, 1)).sum(1)


def plot_logpX_with_ellipses(
        X_centers,
        H_per_x,
        xlim,
        ylim,
        grid_n=201,
        ellipse_scale=0.3,
        step_centers=2,
        savepath="gmm_ellipses.pdf",
):
    x1 = np.linspace(xlim[0], xlim[1], grid_n)
    x2 = np.linspace(ylim[0], ylim[1], grid_n)
    XX, YY = np.meshgrid(x1, x2)
    grid = torch.tensor(np.stack([XX, YY], axis=-1), device=device, dtype=dtype).reshape(-1, 2)
    with torch.no_grad():
        Z = log_pX(grid).reshape(grid_n, grid_n).cpu().numpy()
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("none")
    ax.contourf(XX, YY, Z, levels=20, cmap=plt.cm.viridis, linewidths=0.0, antialiased=True)
    Xc = X_centers.to(device=device, dtype=dtype)
    Hc = H_per_x.to(device=device, dtype=dtype)
    if step_centers > 1:
        Xc = Xc[::step_centers]
        Hc = Hc[::step_centers]
    Xc_np, Hc_np = Xc.cpu().numpy(), Hc.cpu().numpy()
    for (cx, cy), H in zip(Xc_np, Hc_np):
        lam, V = np.linalg.eigh(H)
        r = ellipse_scale / np.sqrt(lam)
        i = int(np.argmax(r))
        j = 1 - i
        ang = np.degrees(np.arctan2(V[1, i], V[0, i]))
        ax.add_patch(Ellipse((cx, cy), width=2 * r[i], height=2 * r[j], angle=ang,
                             fill=False, edgecolor="black", linewidth=0.3))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_zorder(10000)
    plt.tight_layout()
    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
    plt.savefig(savepath)
    plt.close(fig)


if __name__ == "__main__":
    grid_size = 15
    n = 4
    gx = torch.linspace(-width * n, width * n, grid_size, device=device, dtype=dtype)
    gy = torch.linspace(-height * n, height * n, grid_size, device=device, dtype=dtype)
    gxx, gyy = torch.meshgrid(gx, gy, indexing="ij")
    X_centers = torch.stack([gxx, gyy], dim=-1).reshape(-1, 2)
    xlim = (-n * width, n * width)
    ylim = (-n * height, n * height)
    gammas = torch.logspace(-4, 4, 200, device=device, dtype=dtype, base=2)
    H_per_x = compute_H_per_x(X_centers, gammas, num_noises=50, seed=123)
    plot_logpX_with_ellipses(
        X_centers, H_per_x,
        xlim=xlim, ylim=ylim, grid_n=200,
        ellipse_scale=0.3, step_centers=2,
        savepath="gmm_ellipses.pdf"
    )
