import torch
import numpy as np
import torch.nn as nn
from .module import UnitRefEncoder, WNBlock


def swish(x):
    return x*torch.sigmoid(x)


class Model(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.ts_channels = h.ts_channels
        self.cfg_p = h.cfg_p
        self.cfg_s = h.cfg_s if h.cfg_p > 0 else 0
        self.cfg = nn.Parameter(torch.randn(1, h.ure_channels, 1))

        self.fc_t1 = nn.Linear(h.ts_channels, h.ts_channels)
        self.fc_t2 = nn.Linear(h.ts_channels, h.wn_channels)
        self.x_conv = nn.Conv1d(h.num_mels, h.wn_channels, 1)

        self.URE = UnitRefEncoder(h)
        self.WN = WNBlock(h)

        self.apply_weight_norm()

    def forward(self, x_t, ts, c, ref):
        ts = step_embed(ts, self.ts_channels)
        ts = swish(self.fc_t1(ts))
        ts = swish(self.fc_t2(ts)).unsqueeze(2)
        x_t = self.x_conv(x_t)+ts

        if self.training:
            cfg_m = torch.rand((c.size(0), 1, 1), device=c.device) < self.cfg_p
            c = torch.where(cfg_m, self.cfg.expand_as(c), c)
            eps = self.WN(x_t, c, ref)
        else:
            assert c.size(0) == 1
            x_t = x_t.expand(2, -1, -1)
            c = torch.cat([c, self.cfg.expand_as(c)], 0)
            ref = ref.expand(2, -1, -1)
            eps = self.WN(x_t, c, ref)
            eps = torch.chunk(eps, 2, dim=0)
            eps = (1+self.cfg_s)*eps[0]-self.cfg_s*eps[1]

        return eps

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError: # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)


def step_embed(ts, ts_channels):
    '''
    ts: (B, 1)
    ts_channels: int
    return: (B, ts_channels)
    '''
    assert ts_channels%2 == 0
    half_dim = ts_channels//2
    _embed = np.log(10000)/(half_dim-1)
    _embed = torch.exp(torch.arange(half_dim)*-_embed).to(ts.device)
    ts = ts*_embed
    ts = torch.cat((torch.sin(ts), torch.cos(ts)), 1)

    return ts
