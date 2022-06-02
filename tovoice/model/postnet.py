from torch import nn
from config import CFG


class Postnet(nn.Module):

    def __init__(self):
        super(Postnet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Conv1d(512, 80, kernel_size=5, padding=2),
            nn.BatchNorm1d(80),
        )

    def forward(self, x):
        return self.conv(x)
