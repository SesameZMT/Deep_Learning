import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

# Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # select cuda or cpu
print('Using PyTorch version:', torch.__version__, ' Device:', device)
# Hyper-parameters 

batch_size = 64
learning_rate = 0.001

# MNIST dataset
train_dataset = datasets.MNIST(
    root='MLP/data', # url of the data
    train=True,
    transform = transforms.ToTensor(), 
    download=True
)
test_dataset = datasets.MNIST(
    root='MLP/data',
    train=False, # is not train data
    transform=transforms.ToTensor() # work on imput and change the dimension
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True # shuffle the data to be random
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

model = MLPMixer(
    image_size = 28,
    channels = 1,
    patch_size = 7,
    dim = 14,
    depth = 3,
    num_classes = 10
)
model.to(device)

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # default optimizer
mse = nn.MSELoss()
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size_train = data.shape[0]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pre_out = model(data)
        targ_out = torch.nn.functional.one_hot(target,num_classes=10)
        targ_out = targ_out.view((batch_size_train,10)).float()
        loss = mse(pre_out, targ_out)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss =0
    with torch.no_grad():
        for data, target in test_loader:
            batch_size_test = data.shape[0]
            data, target = data.to(device), target.to(device)
            pre_out = model(data)
            targ_out = torch.nn.functional.one_hot(target,num_classes=10)
            targ_out = targ_out.view((batch_size_test,10)).float()
            test_loss += mse(pre_out, targ_out) # 将一批的损失相加
    
    test_loss /= len(test_loader.dataset)
    print("nTest set: Average loss: {:.4f}".format(test_loss))
n_epochs = 10
    
for epoch in range(n_epochs):               
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)