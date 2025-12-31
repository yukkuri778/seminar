import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, hid1_size = 1024, hid2_size = 512,
                 hid3_size = 256, batch_size = 4):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.b_size = batch_size
        self.fc1 = nn.Linear(784, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, 1)
    
        self.LeakyReLU = nn.LeakyReLU(0.2)