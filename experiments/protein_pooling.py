from torch import nn


class MeanPooling(nn.Module):
    def forward(self, x):
        return x.mean(dim=-1)


class CNNPooling(nn.Module):
    def __init__(self, embed_size, max_length, kernel_size: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.ReflectionPad2d
            nn.Conv1d(embed_size, embed_size, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(max_length),
        )
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x.unsqueeze(1)).squeeze(1)
