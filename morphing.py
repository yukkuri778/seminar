import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils
import os

BASE_DIR = "dcgan_output"
model_save_dir = os.path.join(BASE_DIR, "models")
img_save_dir = os.path.join(BASE_DIR, "images")

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x): return self.main(x)

# モーフィング生成関数
def generate_morphing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 100
    
    model_path = os.path.join(model_save_dir, "generator_final.pth")

    netG = Generator(z_dim=z_dim).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    torch.manual_seed(43)
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    # 参照用グリッド画像を生成・保存
    with torch.no_grad():
        fake_imgs = netG(fixed_noise).detach().cpu()

    grid_img = utils.make_grid(fake_imgs, padding=2, normalize=True)
    
    # グリッド画像保存
    grid_save_path = os.path.join(img_save_dir, "morphing_reference_grid.png")
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Reference Grid (0-63)")
    plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))
    plt.savefig(grid_save_path)
    plt.close()
    
    print(f"参照用画像を保存: {grid_save_path}")
    print("この画像を開いて、変化させたい数字のインデックス(0-63)を確認してください。")
    print("左上:インデックス0、右下:インデックス63")

    #ユーザー入力でインデックス決定
    idx_4_str = input("開始する画像のインデックス : ")
    idx_9_str = input("終了する画像のインデックス : ")
    idx_4 = int(idx_4_str)
    idx_9 = int(idx_9_str)
    print(f"Index {idx_4} から Index {idx_9} への変化を生成します")

    # モーフィング生成
    z_start = fixed_noise[idx_4]
    z_end   = fixed_noise[idx_9]
    steps = 10
    interpolated_images = []

    for i in range(steps):
        alpha = i / (steps - 1)
        z_interp = (1 - alpha) * z_start + alpha * z_end
        z_interp = z_interp.unsqueeze(0)
        with torch.no_grad():
            generated = netG(z_interp).detach().cpu()
            interpolated_images.append(generated)

    final_tensor = torch.cat(interpolated_images, dim=0)
    
    save_path = os.path.join(img_save_dir, f"morphing_{idx_4}_to_{idx_9}.png")
    utils.save_image(final_tensor, save_path, nrow=steps, normalize=True)
    print(f"モーフィング画像を保存: {save_path}")


if __name__ == '__main__':
    generate_morphing()