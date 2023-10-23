import torch
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
    def create_embedding_fn(self):
        embed_fns = []
        # input channel :3
        d = self.kwargs['input_dims']
        out_dim = 0
        # include origin input x
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = self.kwargs['max_freqs_log2']
        N_freqs = self.kwargs['num_freqs']
        if self.kwargs['log_sampling']:
            freq_bands = 2. **torch.linspace(0., max_freq,N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq) )
                out_dim += d
        # point set of embedding function
        self.embed_fns = embed_fns
        self.out_dim =out_dim  # d x 2 x frqs + d
    def embed(self,inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns],-1)
def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,   # whether include origin input
        'input_dims': input_dims,  # embed channels
        'max_freqs_log2': multires-1,  # max embed freqs
        'num_freqs': multires,  # length of embed
        'log_sampling':True,  # samples mothed:log
        'preriodic_fns':[torch.sin, torch.cos]  # periodic function
    }
    # init embedding
    embedder_obj = Embedder(**embed_kwargs)

    # embedding function
    def embed(x, eo=embedder_obj): return eo.embed(x)
    # return embedding function and output channel
    return embed, embedder_obj.out_dim