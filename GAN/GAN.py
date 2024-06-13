import sys
print(sys.version)  # python 3.6
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt

print(torch.__version__)  # 1.0.1

def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0, 2).transpose(0, 1)  # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())

class Discriminator(nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), 784)  # flatten (bs x 1 x 28 x 28) -> (bs x 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out)  # range [-1, 1]
        # convert to image
        out = out.view(out.size(0), 1, 28, 28)
        return out

# Check device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Instantiate a Generator and Discriminator according to their class definition and move to device
D = Discriminator().to(device)
print(D)
G = Generator().to(device)
print(G)

# Load dataset
dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))
                                            ]),
                                            download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Display an example image
ix = 149
x, _ = dataset[ix]
plt.matshow(x.squeeze().numpy(), cmap=plt.cm.gray)
plt.colorbar()

# One image through the discriminator
x_real = x.unsqueeze(0).to(device)  # Add batch dimension and move to device
Dscore = D(x_real)
print(Dscore)

# Batch of images from the dataloader
data_iter = iter(dataloader)
xbatch, _ = next(data_iter)  # 64 x 1 x 28 x 28: minibatch of 64 samples
xbatch = xbatch.to(device)  # Move batch to device
print(xbatch.shape)
print(D(xbatch))  # 64x1 tensor: 64 predictions of probability of input being real
print(D(xbatch).shape)

show_imgs(xbatch)

# Example tensor operations with gradients
x = torch.randn(2, 2, requires_grad=True)
print(x)
print(x.grad)
y = (x**2 + x)
z = y.sum()
print(z)
z.backward()
print(x.grad)
print(2 * x + 1)

# Print gradients of generator parameters
for p in G.parameters():
    print(p.grad)

# Example manual gradient descent
torch.manual_seed(23231)
x1 = torch.Tensor([1, 2, 3, -3, -2])
y = torch.Tensor([3, 6, 9, -9, -6]).view(5, 1)
x2 = torch.randn(5)
x = torch.stack([x1, x2], dim=1)  # 5 x 2 input. 5 datapoints, 2 dimensions.
theta = torch.nn.Parameter(torch.randn(1, 2))
print('x:\n', x)
print('y:\n', y)
print('theta at random initialization: ', theta)
thetatrace = [theta.data.clone()]  # initial value, for logging

ypred = x @ theta.t()  # matrix multiply; (N x 2) * (2 x 1) -> N x 1
print('ypred:\n', ypred)
loss = ((ypred - y)**2).mean()  # mean squared error = MSE
print('mse loss: ', loss.item())
loss.backward()
print('dL / d theta:\n', theta.grad)
theta.data.add_(-0.1 * theta.grad.data)
theta.grad.zero_()
print('theta:\n', theta)
thetatrace.append(theta.data.clone())  # for logging

# Plot SGD optimization trace
thetas = torch.cat(thetatrace, dim=0).numpy()
plt.figure()
plt.plot(thetas[:, 0], thetas[:, 1], 'x-')
plt.plot(3, 0, 'ro')
plt.xlabel('theta[0]')
plt.ylabel('theta[1]')

# Another example with nn.Linear and optimizer
torch.manual_seed(23801)
net = nn.Linear(2, 1, bias=False)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
for i in range(100):
    ypred = net(x)
    loss = ((ypred - y)**2).mean()  # mean squared error = MSE
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(net.weight)

# Re-initialize D, G and move to device
D = Discriminator().to(device)
G = Generator().to(device)
print(D)
print(G)

# Set up optimizers
optimizerD = torch.optim.SGD(D.parameters(), lr=0.03)
optimizerG = torch.optim.SGD(G.parameters(), lr=0.03)
criterion = nn.BCELoss()

lab_real = torch.ones(64, 1, device=device)
lab_fake = torch.zeros(64, 1, device=device)

# For logging
collect_x_gen = []
fixed_noise = torch.randn(64, 100, device=device)
plt.ion()  # Interactive mode on

# Lists to hold discriminator and generator losses for plotting
d_losses = []
g_losses = []

for epoch in range(3):  # 3 epochs
    for i, data in enumerate(dataloader, 0):
        # STEP 1: Discriminator optimization step
        data_iter = iter(dataloader)
        x_real, _ = next(data_iter)
        x_real = x_real.to(device)
        optimizerD.zero_grad()

        D_x = D(x_real)
        lossD_real = criterion(D_x, lab_real)

        z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
        x_gen = G(z).detach()
        D_G_z = D(x_gen)
        lossD_fake = criterion(D_G_z, lab_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # STEP 2: Generator optimization step
        optimizerG.zero_grad()

        z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
        x_gen = G(z)
        D_G_z = D(x_gen)
        lossG = criterion(D_G_z, lab_real)  # -log D(G(z))

        lossG.backward()
        optimizerG.step()

        # Log losses
        d_losses.append(lossD.item())
        g_losses.append(lossG.item())

        if i % 100 == 0:
            x_gen = G(fixed_noise)
            show_imgs(x_gen, new_fig=False)
            plt.draw()
            plt.pause(0.001)  # Pause to update the figure
            print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                epoch, i, len(dataloader), D_x.mean().item(), D_G_z.mean().item()))

    # End of epoch
    x_gen = G(fixed_noise)
    collect_x_gen.append(x_gen.detach().clone())

plt.ioff()  # Turn off interactive mode

# Plot the discriminator and generator losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="G")
plt.plot(d_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fixed_noise = torch.randn(8, 100, device=device)
x_gen = G(fixed_noise)
show_imgs(x_gen)

import random

# 自定义一组随机数
custom_noise = torch.randn(8, 100,device=device)  # 8张图，每张图的随机数维度为100

# 存储每个随机数调整时生成的图像
adjusted_images = []

# 针对自定义的100个随机数，自由挑选5个随机数
selected_indices = random.sample(range(100), 5)

# 调整每个随机数，并生成图像
for idx in selected_indices:
    noise_variations = []
    for i in range(3):
        noise_copy = custom_noise.clone()
        # 在原始随机数的基础上加上一个小的扰动
        noise_copy[:, idx] += 0.1 * (i + 1)
        generated_images = G(noise_copy)
        noise_variations.append(generated_images)
    adjusted_images.append(noise_variations)

# 显示生成的图像并观察变化
for idx, noise_variations in enumerate(adjusted_images):
    print(f"Random number {selected_indices[idx]} adjustments:")
    for i, images in enumerate(noise_variations):
        print(f"Adjustment {i + 1}:")
        show_imgs(images)

plt.show()  # Keep the plot window open
