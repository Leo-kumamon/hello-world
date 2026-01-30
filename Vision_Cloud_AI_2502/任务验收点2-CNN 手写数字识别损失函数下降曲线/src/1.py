import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ==================== 1. 一些基础设置 ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

batch_size = 64
learning_rate = 0.01
num_epochs = 5          # 先跑 5 轮就能有 97% 左右准确率，你可以根据时间增减


# ==================== 2. 数据集和 DataLoader ====================

# 对 MNIST 做的基本变换：转成 tensor，并做 0-1 归一化
transform = transforms.Compose([
    transforms.ToTensor(),                      # [0,255] -> [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 按官方推荐的均值和方差标准化
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==================== 3. 定义 CNN 模型 ====================

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 输入：1x28x28
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # -> 64x7x7
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)   # 10 个类别
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


model = CNN().to(device)
print(model)


# ==================== 4. 损失函数和优化器 ====================

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 用于画损失曲线
loss_list = []      # 每个 batch 的 loss
step_list = []      # 对应的迭代次数（全局 step 计数）
global_step = 0


# ==================== 5. 训练循环 ====================

for epoch in range(num_epochs):
    model.train()   # 进入训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # 1) 前向
        output = model(data)
        loss = criterion(output, target)

        # 2) 反向
        optimizer.zero_grad()
        loss.backward()

        # 3) 更新参数
        optimizer.step()

        global_step += 1

        # 保存 loss，用于画图
        loss_list.append(loss.item())
        step_list.append(global_step)

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")


# ==================== 6. 画损失函数下降曲线 ====================

plt.figure(figsize=(8, 4))
plt.plot(step_list, loss_list)
plt.xlabel("Iteration (global step)")
plt.ylabel("Training Loss")
plt.title("CNN on MNIST - Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()   # 在 PyCharm 里会弹出图形窗口


# ==================== 7. 在测试集上计算准确率 ====================

model.eval()   # 进入测试/推理模式
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        outputs = model(data)
        # dim=1 表示在每一行上取最大值，返回 (max_value, index)
        _, predicted = torch.max(outputs.data, 1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

test_acc = 100 * correct / total
print(f"测试集准确率: {test_acc:.2f}%")
