import torch
import torch.nn as nn
import torch.nn.functional as F
LR_SLOPE = 0.2


class UnitRefEncoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        # unit
        self.embedding = nn.Embedding(h.num_km+1, h.emb_channels, padding_idx=h.num_km)
        self.kmr_conv = nn.Sequential(
            nn.Conv1d(h.emb_channels, h.emb_channels, 3, padding=1),
            nn.LeakyReLU(LR_SLOPE),
            nn.Conv1d(h.emb_channels, h.emb_channels, 3, padding=1),
            nn.LeakyReLU(LR_SLOPE),
            nn.Conv1d(h.emb_channels, h.emb_channels, 3, padding=1),
            nn.LeakyReLU(LR_SLOPE),
            nn.Dropout(0.5)
        )

        # ref
        self.ref_conv = nn.Sequential(
            nn.Conv1d(h.num_mels, h.ref_channels, 3, padding=1),
            nn.LeakyReLU(LR_SLOPE),
            nn.Conv1d(h.ref_channels, h.ref_channels, 3, padding=1),
            nn.LeakyReLU(LR_SLOPE),
            nn.Conv1d(h.ref_channels, h.ref_channels, 3, padding=1),
            nn.LeakyReLU(LR_SLOPE),
            nn.Dropout(0.5)
        )
        self.ref_out = nn.Sequential(
            nn.Conv1d(h.ref_channels*2, h.ref_channels, 1),
            nn.LeakyReLU(LR_SLOPE),
            nn.Conv1d(h.ref_channels, h.ref_channels, 1)
        )

        # combine unit and ref
        self.out = nn.Sequential(
            nn.Conv1d(h.emb_channels+h.ref_channels, h.ure_channels, 1),
            nn.LeakyReLU(LR_SLOPE),
            nn.Conv1d(h.ure_channels, h.ure_channels, 1)
        )

    def forward(self, km, mel):
        # encode km
        km = self.embedding(km).transpose(1, 2)
        km = km+self.kmr_conv(km)

        # encode ref
        ref = self.ref_conv(mel)
        ref = torch.cat([ref.mean(2), ref.std(2)], 1)
        ref = self.ref_out(ref.unsqueeze(2))

        # cat km and ref
        c = self.out(torch.cat([
            km, ref.expand(ref.size(0), ref.size(1), km.size(2))], 1))

        return c, ref


class WNBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.wn_layers = h.wn_layers

        self.x_convs = nn.ModuleList()
        self.c_convs = nn.ModuleList()
        self.ref_convs = nn.ModuleList()
        self.orig_x_convs = nn.ModuleList()
        self.gau_convs = nn.ModuleList()
        self.res_skip_convs = nn.ModuleList()
        for i in range(h.wn_layers):
            self.x_convs.append(
                nn.Conv1d(h.wn_channels, h.wn_channels, 3, padding=1)
            )
            self.c_convs.append(
                nn.Conv1d(h.ure_channels, h.wn_channels, 3, padding=1)
            )
            self.ref_convs.append(
                nn.Conv1d(h.ref_channels, h.wn_channels, 1)
            )
            self.gau_convs.append(
                nn.Conv1d(h.wn_channels, h.wn_channels*2, 3, padding=1)
            )
            self.res_skip_convs.append(
                nn.Conv1d(h.wn_channels, h.wn_channels*2, 1)
            )
        self.out = nn.Sequential(
            nn.Conv1d(h.wn_channels*2, h.wn_channels, 3, padding=1),
            nn.Tanh(),
            nn.Conv1d(h.wn_channels, h.num_mels, 1)
        )

    def forward(self, x_t, c, ref):
        c = F.interpolate(c, size=x_t.size(2))
        skip = None
        for i in range(self.wn_layers):
            y = self.x_convs[i](x_t)+self.c_convs[i](c)+self.ref_convs[i](ref)
            y = torch.chunk(self.gau_convs[i](y), 2, dim=1)
            y = torch.sigmoid(y[0])*torch.tanh(y[1])
            y = torch.chunk(self.res_skip_convs[i](y), 2, dim=1)
            x_t = x_t+y[0]
            skip = y[1] if skip is None else skip+y[1]
        eps = self.out(torch.cat([x_t, skip], 1))

        return eps
