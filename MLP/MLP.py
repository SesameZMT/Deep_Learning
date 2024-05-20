import torch  
import torch.nn as nn
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

# Fully connected neural network with one hidden layer
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28,100) # from input layer to hidden layer
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100,80) # from hidden layer to output layer
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(80, 10)
    def forward(self, x):# connect the layers to net
        out = x.view(-1, 28*28) # reshape the input data
        out = F.relu(self.fc1(x)) # relu activation function
        out = self.fc1_drop(out) # dropout
        out = F.relu(self.fc2(out))
        out = self.fc2_drop(out)
        return F.log_softmax(self.fc3(out), dim=1) # softmax activation function

model = MLPNet().to(device) # three layers of net

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # loss criterion
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # default optimizer

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data= data.reshape(-1, 28*28).to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in test_loader:
        data = data.reshape(-1, 28*28).to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(test_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\ntest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(test_loader.dataset), accuracy))

epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)