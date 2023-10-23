import numpy as np
import torch

from models.embeder import get_embedder
import torch.nn as nn
import torch.nn.functional as F
class NeRF(nn.Module):
    def __init__(self,
                 D=8,  # net depth
                 W=256,  # net width
                 d_in=3,  # sample point (input channel)
                 d_in_view=3,  # unit direction vector
                 multires=0,  # sample encoder length
                 multires_view=0,  # direction  encoder length
                 output_ch=4,  # output channel
                 skips=[4],  # new input position in the net
                 use_viewdirs=False):  # whether use extra direction as input
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        # embeding the samples
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        # embeding the directions
        if multires_view > 0:
            embed_fn, input_ch = get_embedder(multires_view, input_dims=d_in)
            self.embed_fn_view = embed_fn
            self.input_ch_view = input_ch
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # create net
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W+self.input_ch,W) for i in range(D - 1)])
        self.view_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W//2)])
        # extra input views
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)  # volume density
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts,h],-1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.view_linears):
                h = self.view_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,  # input channel
                 d_out,
                 d_hidden,  # depth of hidden layer
                 n_layers,  # net lays
                 skip_in=(4,),  # net skip input
                 multires=0,  # length of embedding
                 bias=0.5,  # bias init
                 scale=1,
                 geometric_init=True,  # init weight
                 weight_norm=True,   # normalization
                 inside_outside=False):
        super(SDFNetwork, self).__init__()
        # net channels
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        # position embedding
        self.embed_fn_fine = None
        if multires > 0:
            # embed function and embed output channel
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        # net layers
        self.num_layers = len(dims)
        # skip input
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers-1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l+1]
                lin = nn.Linear(dims[l], out_dim)

            # whether net weight init manually
            if geometric_init:
                if l == self.num_layers_2:  # last two layer
                    if not inside_outside:
                        # weights satisfy normal distribution
                        torch.nn.init.normal_(lin.weight,mean=np.sqrt(np.pi)/np.sqrt(dims[1]),std=0.0001)
                        # bias set constant
                        torch.nn.init.constant_(lin.bias,-bias)
                    else:
                        # weights satisfy positive distribution
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[1]), std=0.0001)
                        # bias set constant
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0: # first layer
                    # bias set constant
                    torch.nn.init.constant_(lin.bias,0.0)
                    # the first three channel set to constant
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    # the rest channel set to normal distribution
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2)/np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:  # on the new input layer
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2)/np.sqrt(out_dim))
                    # for cat the middle three channel set to constant
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3)], 0.0)

                else:  # the other layer
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2)/np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            # rename the linear layer
            setattr(self, "line" + str(l), lin)
        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        # input 3D position
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "line" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1)/np.sqrt(2)
            x = lin(x)
            if l < self.num_layers -2:
                x = self.activation(x)
        # [batch_sziem, 1+256]  1 means sdf value, 256 mean hidden feature
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    #  get sdf gradient
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class SingleVarianceNetwork:
    def __init__(self):
        print("SingleVarianceNetwork")
        self.parameters = {}


class RenderingNetwork:
    def __init__(self):
        print("RenderingNetwork")
        self.parameters = {}
