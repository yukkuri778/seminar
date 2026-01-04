import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        """
        DCGAN Discriminator (識別器)
        Args:
            nc (int): 入力画像のチャンネル数 (Default: 1)
            ndf (int): フィルタ数の基準 (Default: 64)
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 入力: (nc) x 64 x 64
            # 畳み込みで画像を圧縮していく (鑑定)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
            # 最終的に1つの数字(確率)にする
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
   
    def forward(self, input):
        # 最後に .view(-1) をして、[Batch, 1, 1, 1] を [Batch] の形にする
        return self.main(input).view(-1)