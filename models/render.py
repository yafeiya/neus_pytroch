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

        self.n_samples = n_samples  # coarse samples
        self.m_importance = n_importance  # fine samples
        self.n_outside = n_outside  # background samples
        self.up_sample_steps = up_sample_steps  # upsample scale
        self.perturb = perturb  # destabilization

    def render(self, rays_o, rays_d, near, far, perturb_overwriter= -1, background_rgb=None, cos_anneal_ratio=0.0):
        #  front scene coarse sample
        batch_size = len(rays_o)
        sample_dist = 2.0/self.n_samples   # unit sphere
        # uniform sample
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far-near) * z_vals[None, :]
        # back scene sample
        z_vals_outside = None
        if self.n_outside > None:
            # coarse sample
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_sample = self.n_samples
        perturb = self.perturb
        if perturb_overwriter >= 0:
            perturb = perturb_overwriter
        if perturb > 0:
            # (-0.5,0.5) uniform sample perturb coefficient
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            # perturb on coarse sample
            z_vals = z_vals + t_rand * 2.0 / self.n_samples
            if self.n_outside > 0:
                # get mid
                mids = .5 *(z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                # far samples
                upper = torch.cat([mids,z_vals_outside[..., -1:]], -1)
                # near samples
                lower = torch.cat([z_vals_outside[...,:1],mids], -1)
                # (0,1)uniform sample perturb coefficient
                t_rand = torch.ramd([batch_size,z_vals_outside.shape[-1]])
                # perturb on coarse sample back scene
                z_vals_outside = lower[None,:] + (upper - lower)[None, :] * t_rand
        # sample out of the far
        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside,dim=[-1]) + 1.0 /self.n_samples
