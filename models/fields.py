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
                 nultires_view=0,  # direction  encoder length
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
        if nultires_view > 0:
            embed_fn, input_ch = get_embedder(nultires_view, input_dims=d_in)
            self.embed_fn_view = embed_fn
            self.input_ch_view = input_ch
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # create net
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W+self.input_ch,W) for i in range(D - 1)])

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
        for i,l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)



    def load_state_dict(self):
        print("load checkpoint")


class SDFNetwork:
    def __init__(self):
        print("sdf_network")
        self.parameters = {}


class SingleVarianceNetwork:
    def __init__(self):
        print("SingleVarianceNetwork")
        self.parameters = {}


class RenderingNetwork:
    def __init__(self):
        print("RenderingNetwork")
        self.parameters = {}
