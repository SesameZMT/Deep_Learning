import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 定义稠密层
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)  # 批量归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)  # 卷积层

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))  # 前向传播：归一化 -> 激活 -> 卷积
        out = torch.cat([x, out], 1)  # 将输入和输出拼接在一起
        return out

# 定义稠密块
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))  # 添加稠密层
        self.block = nn.Sequential(*layers)  # 将稠密层按顺序排列

    def forward(self, x):
        return self.block(x)  # 前向传播

# 定义过渡层
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)  # 批量归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)  # 卷积层
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化层

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))  # 前向传播：归一化 -> 激活 -> 卷积
        out = self.pool(out)  # 池化
        return out

# 定义DenseNet模型
class DenseNet(nn.Module):
    def __init__(self, num_blocks, num_layers_per_block, growth_rate, reduction, num_classes):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_channels = 2 * growth_rate  # 初始通道数

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(DenseBlock(num_layers_per_block, num_channels, growth_rate))  # 添加稠密块
            num_channels += num_layers_per_block * growth_rate
            if i != num_blocks - 1:
                out_channels = int(num_channels * reduction)
                self.blocks.append(TransitionLayer(num_channels, out_channels))  # 添加过渡层
                num_channels = out_channels

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)  # 全连接层

    def forward(self, x):
        out = self.pool1(self.relu(self.bn1(self.conv1(x))))  # 初始层前向传播
        for block in self.blocks:
            out = block(out)  # 稠密块前向传播
        out = self.relu(self.bn2(out))  # 最后一次批量归一化和激活
        out = F.adaptive_avg_pool2d(out, (1, 1))  # 自适应平均池化
        out = torch.flatten(out, 1)  # 展平
        out = self.fc(out)  # 全连接层
        return out

# 训练函数
def train(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()  # 设置模型为训练模式
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = F.cross_entropy(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        pred = output.argmax(dim=1, keepdim=True)  # 获取预测值
        correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确预测的数量
        total += target.size(0)
        
        if batch_idx % 100 == 99:
            train_losses.append(loss.item())  # 记录损失
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# 测试函数
def test(model, device, test_loader, test_losses, test_accuracies):
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # 前向传播
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 累计损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测值
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确预测的数量
    test_loss /= len(test_loader.dataset)  # 计算平均损失
    test_losses.append(test_loss)  # 记录损失
    
    accuracy = 100. * correct / len(test_loader.dataset)  # 计算准确率
    test_accuracies.append(accuracy)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# 主函数
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    num_blocks = 3  # 稠密块数量
    num_layers_per_block = 6  # 每个稠密块中的层数
    growth_rate = 12  # 增长率
    reduction = 0.5  # 过渡层中通道数的减少比例
    num_classes = 10  # 分类数量
    batch_size = 64  # 批次大小
    epochs = 30  # 训练轮数
    learning_rate = 0.1  # 学习率

    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  # 训练数据集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练数据加载器

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  # 测试数据集
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试数据加载器

    model = DenseNet(num_blocks, num_layers_per_block, growth_rate, reduction, num_classes).to(device)  # 初始化模型
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)  # 优化器
    print(model)
    train_losses = []  # 记录训练损失
    test_losses = []  # 记录测试损失
    test_accuracies = []  # 记录测试准确率

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, train_losses)  # 训练模型
        test(model, device, test_loader, test_losses, test_accuracies)  # 测试模型

    epochs_range = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Train Loss')
    plt.legend(loc='best')
    plt.title('Train Loss')

    plt.figure()
    epochs_range = np.arange(1, epochs + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.legend(loc='best')
    plt.title('Test Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.show()

    print("Training and testing completed.")
    
if __name__ == '__main__':
    main()  # 运行主函数
