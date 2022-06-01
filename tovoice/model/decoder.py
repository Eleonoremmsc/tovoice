from torch import nn
from config import CFG


class Decoder(nn.Module):
    def __init__(self):

        super(Decoder, self).__init__()

        self.lstm1 = nn.LSTM(
            CFG.dim_neck * 2 + CFG.dim_emb, CFG.dim_pre, 1, batch_first=True
        )

        self.conv = nn.Sequential(
            nn.Conv1d(CFG.dim_pre, CFG.dim_pre, kernel_size=5, padding=2),
            nn.BatchNorm1d(CFG.dim_pre),
            nn.ReLU(),
            nn.Conv1d(CFG.dim_pre, CFG.dim_pre, kernel_size=5, padding=2),
            nn.BatchNorm1d(CFG.dim_pre),
            nn.ReLU(),
            nn.Conv1d(CFG.dim_pre, CFG.dim_pre, kernel_size=5, padding=2),
            nn.BatchNorm1d(CFG.dim_pre),
            nn.ReLU(),
        )

        self.lstm2 = nn.LSTM(CFG.dim_pre, 1024, 2, batch_first=True)

        self.linear = nn.Linear(1024, 80)

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        x = self.conv(x)
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)

        decoder_output = self.linear(outputs)

        return decoder_output
