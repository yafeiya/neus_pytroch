import torch


class NeuSRender:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network

        self.n_samples = n_samples  # rough samples
        self.m_importance = n_importance  # fine samples
        self.n_outside = n_outside  # background samples
        self.up_sample_steps = up_sample_steps  # upsample scale
        self.perturb = perturb  # destabilization

    def render(self, rays_o, rays_d, near, far, perturb_overwriter= -1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0/self.n_samples   # unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far-near) * z_vals[None, :]