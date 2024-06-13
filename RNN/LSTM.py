from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import unicodedata
import string

# 查找所有名字文件
def findFiles(path): return glob.glob(path)

# 读取文件内容
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# 将Unicode字符串转换为ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# 构建类别字典
def buildCategoryLines():
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    return category_lines, all_categories

# 将字母转换为张量
def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# 随机选择一个类别和一个样本
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# 自定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        # 定义LSTM的四个门的权重和偏置
        self.W_f = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))  # 遗忘门的权重
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))                            # 遗忘门的偏置

        self.W_i = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))  # 输入门的权重
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))                            # 输入门的偏置

        self.W_C = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))  # 候选细胞状态的权重
        self.b_C = nn.Parameter(torch.Tensor(hidden_size))                            # 候选细胞状态的偏置

        self.W_o = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))  # 输出门的权重
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))                            # 输出门的偏置

        # 定义线性层
        self.linear = nn.Linear(hidden_size, output_size)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 仅对权重进行Kaiming初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        seq_length = input_seq.size(1)

        # 初始化隐状态和细胞状态为零
        h_t = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
        C_t = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)

        for t in range(seq_length):
            x_t = input_seq[:, t, :]
            x_t = x_t.squeeze(1)  # 调整 x_t 的维度使其与 h_t 一致
            combined = torch.cat((x_t, h_t), dim=1)  # 将当前输入和前一时间步的隐状态拼接

            # 计算遗忘门
            f_t = torch.sigmoid(combined @ self.W_f.t() + self.b_f)
            # 计算输入门
            i_t = torch.sigmoid(combined @ self.W_i.t() + self.b_i)
            # 计算候选细胞状态
            C_tilde_t = torch.tanh(combined @ self.W_C.t() + self.b_C)
            # 计算输出门
            o_t = torch.sigmoid(combined @ self.W_o.t() + self.b_o)

            # 更新细胞状态
            C_t = f_t * C_t + i_t * C_tilde_t
            # 更新隐状态
            h_t = o_t * torch.tanh(C_t)

        # 通过线性层生成输出
        output = self.linear(h_t)
        return output

# 辅助函数
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# 绘制损失曲线
def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# 绘制热力图
def plot_confusion_matrix(confusion_matrix, all_categories):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix.numpy())
    fig.colorbar(cax)

    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

# 绘制准确率曲线
def plot_accuracy(accuracies):
    plt.figure()
    plt.plot(accuracies)
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

# 主函数
if __name__ == "__main__":
    # 字母表和类别
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    category_lines, all_categories = buildCategoryLines()
    n_categories = len(all_categories)

    # 初始化模型
    n_hidden = 128
    model = LSTM(n_letters, n_hidden, n_categories)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    n_iters = 10000
    print_every = 500
    plot_every = 100

    current_loss = 0
    all_losses = []
    accuracies = []
    confusion = torch.zeros(n_categories, n_categories)

    start = time.time()
    print(model)

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        category_tensor = category_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        line_tensor = line_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        optimizer.zero_grad()
        output = model(line_tensor.unsqueeze(0))
        loss = criterion(output, category_tensor)
        loss.backward()
        optimizer.step()

        current_loss += loss.item()

        # 计算准确率
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

        # 打印进度
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            category_i = all_categories.index(category)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # 记录损失
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

            # 计算并记录准确率
            total = confusion.sum().item()
            correct = confusion.trace().item()
            accuracy = correct / total
            accuracies.append(accuracy)

    # 绘制损失曲线
    plot_loss(all_losses)

    # 绘制准确率曲线
    plot_accuracy(accuracies)

    # 归一化混淆矩阵
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # 绘制混淆矩阵
    plot_confusion_matrix(confusion, all_categories)
