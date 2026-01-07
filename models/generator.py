import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=1):
        """
        DCGAN Generator (生成器)
        Args:
            z_dim (int): 入力ノイズの次元 (default: 100)
            ngf (int): ジェネレータのフィルタ数の基準 (default: 64)
            nc (int): 出力画像のチャンネル数 (MNISTなら1, カラーなら3)
        """
        super(Generator, self).__init__()
        
        # ネットワーク構造の定義
        self.main = nn.Sequential(
            # 入力: (z_dim) x 1 x 1
            nn.ConvTranspose2d(z_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 出力: (nc) x 32 x 32
        )

    def forward(self, x):
        return self.main(x)    