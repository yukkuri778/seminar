from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloader(
    batch_size=128,
    train=True,
    data_root="../data/MNIST",
    shuffle=True,
    drop_last=True
):
    """
    DCGAN 用 MNIST DataLoader
    Generator の tanh 出力に合わせて [-1, 1] に正規化
    """

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),                 # [0,1]
        transforms.Normalize((0.5,), (0.5,))   # → [-1,1]
    ])

    dataset = datasets.MNIST(
        root=data_root,
        train=train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    return dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)