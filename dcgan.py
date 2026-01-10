# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

#出力フォルダの構成
BASE_DIR = "dcgan_output"
img_save_dir = os.path.join(BASE_DIR, "images")
model_save_dir = os.path.join(BASE_DIR, "models")
graph_save_dir = os.path.join(BASE_DIR, "graphs")

# フォルダが存在しなければ作成
os.makedirs(img_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(graph_save_dir, exist_ok=True)

# データセット読み込み
def get_mnist_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 重み初期化関数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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

#Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): return self.main(x).view(-1)

#学習処理
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    print(f"画像保存先: {img_save_dir}")

    # ハイパーパラメータ
    batch_size = 128 #バッチサイズ
    z_dim = 100 #潜在変数の次元
    lr = 0.0002 #学習率
    epochs = 20 #エポック数

    dataloader = get_mnist_dataloader(batch_size)
    netG = Generator(z_dim=z_dim).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    epoch_G_losses = []
    epoch_D_losses = []

    print("学習開始")

    for epoch in range(epochs):
        running_d_loss = 0.0
        running_g_loss = 0.0

        for i, data in enumerate(dataloader):
            netD.zero_grad()
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), 0.9, dtype=torch.float, device=device) 

            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_data = netG(noise)
            label.fill_(0.)

            output = netD(fake_data.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            running_d_loss += (errD_real.item() + errD_fake.item())

            netG.zero_grad()
            label.fill_(1.)
            output = netD(fake_data)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            running_g_loss += errG.item()

        # 平均Loss
        avg_d_loss = running_d_loss / len(dataloader)
        avg_g_loss = running_g_loss / len(dataloader)
        epoch_D_losses.append(avg_d_loss)
        epoch_G_losses.append(avg_g_loss)

        print(f'[Epoch {epoch+1}/{epochs}] Loss_D: {avg_d_loss:.4f} Loss_G: {avg_g_loss:.4f}')

        # 画像保存
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        
        # epochごとの画像を保存
        save_path = os.path.join(img_save_dir, f'epoch_{epoch+1}.png')
        utils.save_image(fake, save_path, normalize=True)

        # モデルを保存
        torch.save(netG.state_dict(), os.path.join(model_save_dir, "generator_final.pth"))

    print("学習完了")
    
    # Lossグラフ作成・保存
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss")
    x_axis = range(1, epochs + 1)
    plt.plot(x_axis, epoch_G_losses, label="Generator Loss", marker='o')
    plt.plot(x_axis, epoch_D_losses, label="Discriminator Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(graph_save_dir, "loss_graph.png"))
    plt.close() 
    print(f"Lossグラフを保存: {graph_save_dir}/loss_graph.png")

if __name__ == '__main__':
    train()