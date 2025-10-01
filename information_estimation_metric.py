import math
import torch
import k_diffusion as kd

torch.set_float32_matmul_precision('high')


class DiffusionModelCheckpointWrapper(torch.nn.Module):
    def __init__(self,
                 k_diffusion_model_ckpt_path: str,
                 k_diffusion_model_dtype: str,
                 k_diffusion_model_compile: bool = True):
        super().__init__()
        self.denoiser_model, self.config = kd.utils.load_k_diffusion_model(k_diffusion_model_ckpt_path,
                                                                           k_diffusion_model_dtype,
                                                                           k_diffusion_model_compile)

        self.input_image_size = int(self.config['model']['input_size'][0])
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, y_sigma: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        assert y_sigma.shape[2] == y_sigma.shape[3] == self.input_image_size
        return self.denoiser_model(y_sigma, sigma)


def learned_f_iem(z_gamma, dqv, learned_f, alpha):
    integrand = learned_f(alpha * z_gamma.transpose(0, 1)).transpose(0, 1).pow(2) * dqv
    return torch.sum(integrand, dim=0)


def square_f_iem(z_gamma, dqv, alpha):
    integrand = (2 * alpha * z_gamma).pow(2) * dqv
    return torch.sum(integrand, dim=0)


def standard_iem(dqv):
    return torch.sum(dqv, dim=0)


class InformationEstimationMetric(torch.nn.Module):
    def __init__(self,
                 diffusion_model_ckpt_path,
                 diffusion_model_dtype,
                 diffusion_model_compile: bool = True):
        super().__init__()
        self.diffusion_model = DiffusionModelCheckpointWrapper(diffusion_model_ckpt_path,
                                                               diffusion_model_dtype,
                                                               diffusion_model_compile)

        # We perform cumsum in matrix form to improve precision.
        self.cumsum_mat = None
        self.cum_z_gamma = None

    def _get_cumsum_mat(self, num_gamma: int, device) -> torch.Tensor:
        cumsum_mat = self.cumsum_mat
        if (cumsum_mat is None) or (cumsum_mat.shape[0] != num_gamma) or (cumsum_mat.device != device):
            self.cumsum_mat = cumsum_mat = torch.tril(
                torch.ones((num_gamma, num_gamma), device=device, dtype=torch.float64))
        return cumsum_mat

    def _cumsum(self, x: torch.Tensor) -> torch.Tensor:
        cumsum_mat = self._get_cumsum_mat(x.shape[0], x.device)
        return cumsum_mat @ x.to(torch.float64)

    def get_loguniform_schedule(self, num_noises, bs, device, min_value, max_value):
        min_value = math.log(min_value)
        max_value = math.log(max_value)
        u = torch.linspace(0.0, 1.0, num_noises, device=device).reshape(num_noises, 1).expand(-1, bs)
        return (u * (max_value - min_value) + min_value).exp().flip(dims=(0,))

    def get_brownian_motion(self, x, num_gamma, sigma_min=1e-3, sigma_max=1e3, seed=None):
        device, dtype = x.device, x.dtype

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        sigma = self.get_loguniform_schedule(num_gamma + 1,
                                             x.shape[0],
                                             device=device,
                                             min_value=sigma_min,
                                             max_value=sigma_max)
        gamma = 1 / sigma.pow(2)
        dgamma = gamma[1:] - gamma[:-1]
        assert torch.all(dgamma >= 0)

        dw = (torch.randn(num_gamma, *x.shape, device=device, dtype=torch.float64, generator=generator)
              * kd.utils.append_dims(dgamma.sqrt(), x.ndim + 1))

        w0 = (torch.randn(1, *x.shape, device=device, dtype=torch.float64, generator=generator)
              * kd.utils.append_dims(gamma[0].to(torch.float64).sqrt().unsqueeze(0), x.ndim + 1))
        w = torch.cat((w0, w0 + torch.cumsum(dw, dim=0)), dim=0)

        return w[:-1].to(dtype), dw.to(dtype), gamma, dgamma

    def get_sde_elements(self, x1, x2, w, dw, gamma, dgamma):
        gamma = kd.utils.append_dims(gamma[:-1], x1.ndim + 1)
        sigma = 1.0 / gamma.sqrt()

        y1_gamma = x1.unsqueeze(0) + w / gamma
        y2_gamma = x2.unsqueeze(0) + w / gamma
        num_gamma, bs = y1_gamma.shape[0], y1_gamma.shape[1]

        y_gamma = torch.cat((y1_gamma, y2_gamma), dim=1).reshape(2 * num_gamma * bs, *x1.shape[1:])
        sigma_expanded = sigma.reshape(num_gamma, bs)
        sigma_expanded = torch.cat([sigma_expanded, sigma_expanded], dim=1).reshape(2 * num_gamma * bs)

        x12_pred = self.diffusion_model(y_gamma, sigma_expanded).to(torch.float32).reshape(num_gamma, 2 * bs, -1)
        x1_pred, x2_pred = x12_pred.chunk(2, dim=1)
        x1_pred, x2_pred = x1_pred.reshape(num_gamma, bs, *x1.shape[1:]), x2_pred.reshape(num_gamma, bs, *x2.shape[1:])

        eps1_gamma = (x1.unsqueeze(0) - x1_pred).flatten(start_dim=2)
        eps2_gamma = (x2.unsqueeze(0) - x2_pred).flatten(start_dim=2)

        diffusion_coef = eps1_gamma - eps2_gamma
        stochastic_increment = torch.einsum('ijk,ijk->ij', diffusion_coef, dw.flatten(start_dim=2))

        drift_coef = (eps1_gamma.pow(2) - eps2_gamma.pow(2)).sum(dim=2)
        drift_increment = 0.5 * drift_coef * dgamma.view(num_gamma, bs)

        dz_gamma = drift_increment + stochastic_increment
        z_gamma = self._cumsum(dz_gamma)
        z_gamma = torch.cat((torch.zeros_like(z_gamma[:1]), z_gamma), dim=0)[:-1]

        dqv = (diffusion_coef.pow(2).sum(dim=2) * dgamma.view(num_gamma, bs))

        return {
            'z_gamma': z_gamma,
            'dqv': dqv,
            'x1_pred': x1_pred,
            'x2_pred': x2_pred,
        }

    @torch.inference_mode()
    def forward(self,
                x1,
                x2,
                num_gamma,
                sigma_min,
                sigma_max,
                iem_type,
                learned_f=None,
                seed=None):
        # Expects x1 and x2 in [-1, 1]

        w, dw, gamma, dgamma = self.get_brownian_motion(x1,
                                                        num_gamma,
                                                        sigma_min=sigma_min,
                                                        sigma_max=sigma_max,
                                                        seed=seed)

        assert gamma.ndim == 2 and gamma.shape[0] == num_gamma + 1 and gamma.shape[1] == x1.shape[0]

        sde_out = self.get_sde_elements(x1, x2, w, dw, gamma, dgamma)

        z_gamma = sde_out['z_gamma']
        dqv = sde_out['dqv']

        # alpha = 1 / d, where d is the dimensionality of the signal.
        alpha = 1 / x1[0].numel()

        if iem_type == 'learned_f':
            assert learned_f is not None
            iem = learned_f_iem(z_gamma, dqv, learned_f, alpha)
        elif iem_type == 'square_f':
            iem = square_f_iem(z_gamma, dqv, alpha)
        elif iem_type == 'standard':
            iem = standard_iem(dqv)
        else:
            raise NotImplementedError(f"iem_type {iem_type} not implemented.")
        return iem.sqrt()
