import torch
from torch import nn
from config import CFG


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(80 + CFG.dim_emb, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(512, CFG.dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, spc, emb):  # 1 spec, 1 Embedding

        # Preprocessing and concatenating
        spc = spc.squeeze(1).transpose(2, 1)
        emb = emb.unsqueeze(-1).expand(-1, -1, spc.size(-1))
        spc = torch.cat((spc, emb), dim=1)

        # Convolution layers
        spc = self.conv(spc)
        spc = spc.transpose(1, 2)

        # LSTM layer
        outputs, _ = self.lstm(spc)

        # Down 1 and Down 2
        out_forward = outputs[:, :, : CFG.dim_neck]
        out_backward = outputs[:, :, CFG.dim_neck :]

        # TODO Clarify
        codes = []
        for i in range(0, outputs.size(1), CFG.freq):
            codes.append(
                torch.cat(
                    (out_forward[:, i + CFG.freq - 1, :], out_backward[:, i, :]),
                    dim=-1,
                )
            )

        # here pre-bottleneck and post-bottleneck are concatenated
        # concat version is appended x times in codes
        # x : (from 0 to (length of outputs - axis 1)) with step : freq

# concatenates forward(i + CFG.freq - 1) and backward(i), i = "columns"
# -> only concatenates slices of tensor -> abh von freq = 16
# ex.: append concat of slice(0+16-1) and slice(0) (along dimension 1)
#      append concat of slice(17+16-1) and slice(16) (along dimension 1)

# codes is a list of

        return codes

"""
as a key step of constructing the information bottleneck,
both the forward and backward outputs of the bidirectional
LSTM are downsampled by 32
downsampled differently for back- and forward
forward output, the time steps {0, 32, 64, · · · }
backward output, the time steps {31, 63, 95, · · · }
"""
