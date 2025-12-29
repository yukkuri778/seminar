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
