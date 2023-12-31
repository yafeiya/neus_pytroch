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
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W+self.input_ch, W) for i in range(D - 1)])
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
            print("第一个nerf(embed_fn)输入位置size：", input_pts.size())
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)
            print("第一个nerf(embed_fn_view)输入方向size：", input_views.size())
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
        self.num_layers = len(dims) # 10
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
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)



class RenderingNetwork:
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


keys = {"d_out": 257,
        "d_in":3,
        "d_hidden" : 256, 
        "n_layers" : 8,
        "skip_in" : [4],
        "multires" :6,
        "bias" : 0.5,
        "geometric_init" : True}
 # NCHW
 # pts Nx3 dir Nx3
if __name__ == "__main__":
    model = NeRF(use_viewdirs=True).cuda()
    pts = torch.randn(256,3).cuda()  
    dir = torch.randn(256,3).cuda()  # NCHW
    r,y = model(pts,dir)
    for i in range(7):
        print(i)
    print(y.size(),r.size())
    a = 2. ** torch.linspace(0.,5, 6)
    print(a)