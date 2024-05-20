import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

loss_train = []
loss_test = []
acc = []

# BasicBlock for ResNet-18 and ResNet-34
class BasicBlock(nn.Module):
    expansion = 1  # Expansion factor for output channels

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # Save the input as identity
        if self.downsample is not None:
            identity = self.downsample(x)  # Apply downsample if needed
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Add identity (skip connection)
        out = self.relu(out)
        return out

# Bottleneck block for ResNet-50, ResNet-101, and ResNet-152
class Bottleneck(nn.Module):
    expansion = 4  # Expansion factor for output channels

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # Save the input as identity
        if self.downsample is not None:
            identity = self.downsample(x)  # Apply downsample if needed
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity  # Add identity (skip connection)
        out = self.relu(out)
        return out

# ResNet class to construct the network
class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.include_top:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
        return out

# Helper functions to create specific ResNet architectures
def ResNet18(classes_num=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], classes_num)

def ResNet34(classes_num=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], classes_num)

def ResNet50(classes_num=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], classes_num)

def ResNet101(classes_num=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], classes_num)

def ResNet152(classes_num=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], classes_num)

# Training function
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # Print log every 100 batches
            temp = running_loss / 100
            loss_train.append(temp)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {running_loss / 100:.6f}')
            running_loss = 0.0

# Testing function
def test(model, device, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Forward pass
            test_loss += criterion(output, target).item()  # Accumulate test loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
    test_loss /= len(test_loader.dataset)  # Compute average loss
    loss_test.append(test_loss)
    temp = 100. * correct / len(test_loader.dataset)
    acc.append(temp)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def main():
    # 检查GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载训练集和测试集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)
    
    # # 初始化ResNet-18模型
    # model = ResNet18(classes_num=10).to(device)
    # print(model)
    
    # 初始化ResNet-34模型
    model = ResNet34(classes_num=10).to(device)
    print(model)
    
    # # 初始化ResNet-50模型
    # model = ResNet50(classes_num=10).to(device)
    # print(model)
    
    # # 初始化ResNet-101模型
    # model = ResNet101(classes_num=10).to(device)
    # print(model)
    
    # # 初始化ResNet-152模型
    # model = ResNet152(classes_num=10).to(device)
    # print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # 训练和测试模型
    epoch_num = 30
    for epoch in range(1,epoch_num+1):  # 训练10个epoch
        train(model, device, train_loader, criterion, optimizer, epoch)
        test(model, device, test_loader, criterion)
    
    epoch = np.arange(1, 3*epoch_num+1)
    # 绘制loss曲线图
    plt.figure()
    plt.plot(epoch, loss_train, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    epoch = np.arange(1, epoch_num+1)
    plt.figure()
    plt.plot(epoch, loss_test, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()
    
    # 绘制accuracy曲线图
    plt.figure()
    plt.plot(epoch, acc, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()

    plt.show()
    
    # 保存模型
    torch.save(model.state_dict(), "resnet18_cifar10.pth")

if __name__ == '__main__':
    main()