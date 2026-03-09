import math
import torch
from torch import distributions as D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    return LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )

trunc_cmap  = truncate_colormap(plt.cm.get_cmap("viridis"), 0, 1)
trunc_cmap2 = truncate_colormap(plt.cm.get_cmap("Greys"), 1, 1)

sns.set_theme(palette="cividis", style="whitegrid", font_scale=1, rc={
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.color": "black", "ytick.color": "black",
    "legend.labelcolor": "black", "pgf.texsystem": "pdflatex",
    'font.family': 'serif', 'font.size': 7, 'text.usetex': True,
    'pgf.rcfonts': True,
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{amsmath}\usepackage{bm}\usepackage{mathtools}\usepackage{dsfont}\usepackage{amssymb}\usepackage{pifont}\newcommand{\cmark}{\ding{51}}\newcommand{\xmark}{\ding{55}}'
})

means   = torch.tensor([[0., 1.], [1., -1.]])
covs    = torch.tensor([[[1., 0.], [0., 0.1]], [[1., 0.5], [0.5, 0.4]]])
weights = torch.tensor([0.3, 0.7])
log_w   = torch.log_softmax(weights.log().clamp_min(1e-32), dim=-1)

def _mvnormal_log_prob(y, mean, cov):
    d = y.shape[-1]
    chol = torch.linalg.cholesky(cov)
    diff = y - mean
    z = torch.cholesky_solve(diff.unsqueeze(-1), chol).squeeze(-1)
    logdet = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet + (diff * z).sum(-1))

def log_pX(x):
    xK = x.unsqueeze(-2).expand(*x.shape[:-1], means.shape[0], 2)
    return torch.logsumexp(_mvnormal_log_prob(xK, means, covs) + log_w, dim=-1)

def get_Y_distribution(gamma):
    new_means = gamma * means
    new_covs  = gamma**2 * covs + torch.eye(2) * gamma
    return D.MixtureSameFamily(D.Categorical(weights),
                               D.MultivariateNormal(new_means, new_covs))

def log_p_Y_given_X(y, x, gamma):
    return D.MultivariateNormal(gamma * x, torch.eye(2) * gamma).log_prob(y)

def log_p_Y(y, gamma):
    return get_Y_distribution(gamma).log_prob(y)

def score_diff_y(y, x, gamma):
    if float(gamma) == 0.0:
        prior_mean = (weights.unsqueeze(1) * means).sum(0).expand(x.shape[0], -1)
        return x - prior_mean
    y = y.clone().detach().requires_grad_(True)
    g1 = torch.autograd.grad(log_p_Y_given_X(y, x, gamma).sum(), y, create_graph=True)[0]
    g2 = torch.autograd.grad(log_p_Y(y, gamma).sum(), y, create_graph=True)[0]
    return g1 - g2

def compute_quadratic_variation(x1, x2, W, gammas):
    num_eps = W.shape[1]
    delta_gamma = gammas[1:] - gammas[:-1]
    x1 = x1.view(1, 1, x1.shape[0], 2).expand(-1, num_eps, -1, -1)
    x2 = x2.view(1, 1, x2.shape[0], 2).expand(-1, num_eps, -1, -1)
    y1_path = gammas.view(-1, 1, 1, 1) * x1 + W
    y2_path = gammas.view(-1, 1, 1, 1) * x2 + W

    qv = [torch.zeros(num_eps, max(x1.shape[2], x2.shape[2]))]
    for i, gamma in enumerate(gammas[:-1]):
        y1 = y1_path[i].reshape(num_eps * x1.shape[2], 2)
        y2 = y2_path[i].reshape(num_eps * x2.shape[2], 2)
        s1 = score_diff_y(y1, x1.reshape(num_eps * x1.shape[2], 2), gamma).reshape(num_eps, 1, 2)
        s2 = score_diff_y(y2, x2.reshape(num_eps * x2.shape[2], 2), gamma).reshape(num_eps, -1, 2)
        qv.append(qv[-1] + (s1 - s2).pow(2).sum(2) * delta_gamma[i])

    return torch.stack(qv, dim=0).detach().cpu(), gammas

def make_ax(width, height):
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    ax.set_facecolor('none')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_zorder(10000)
    return fig, ax

if __name__ == "__main__":
    x1_sample = torch.tensor([[2.5, -2.]])
    width, height, lim = 1.2, 0.8, 4
    grid_size = 200
    x = np.linspace(-width * lim, width * lim, grid_size)
    y = np.linspace(-height * lim, height * lim, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = torch.tensor(np.stack([xx, yy], -1).reshape(-1, 2), dtype=torch.float32)

    num_gamma, num_eps = 200, 50
    gammas = torch.logspace(-10, 10, num_gamma, base=2)
    delta_gamma = gammas[1:] - gammas[:-1]
    torch.manual_seed(42)
    dW = torch.randn(num_gamma - 1, num_eps, 1, 2) * delta_gamma.sqrt().reshape(-1, 1, 1, 1)
    W = torch.zeros(num_gamma, num_eps, 1, 2)
    W[1:] = torch.cumsum(dW, 0)

    qv, gammas = compute_quadratic_variation(x1_sample, grid_points, W, gammas)
    qv = qv[-1].mean(0).reshape(grid_size, grid_size)

    with torch.no_grad():
        log_probs_pX = log_pX(grid_points).reshape(grid_size, grid_size).numpy()

    scatter_kw = dict(s=50, color='white', marker='*', label=r'$\bm{x}_{\text{ref.}}$',
                      edgecolor='black', linewidth=0.5, zorder=10)
    clabel_kw  = dict(fmt="%.1f", fontsize=4, inline=True, inline_spacing=0.5)

    fig, ax = make_ax(width, height)
    ax.contourf(xx, yy, log_probs_pX, levels=20, cmap=trunc_cmap, linewidths=0.0, antialiased=True)
    vmin, vmax = qv.sqrt().min().item(), qv.sqrt().max().item()
    cf = ax.contour(xx, yy, qv.sqrt().numpy(), levels=np.linspace(0, int(np.ceil(vmax)), 15),
                    cmap=trunc_cmap2, vmin=vmin, vmax=vmax, linewidths=0.3)
    for txt in ax.clabel(cf, **clabel_kw):
        txt.set_clip_path(ax.patch)
    ax.scatter(x1_sample[0, 0], x1_sample[0, 1], **scatter_kw)
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig('gmm_global.pdf')
