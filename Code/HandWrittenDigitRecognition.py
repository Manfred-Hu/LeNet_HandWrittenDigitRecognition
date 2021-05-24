# CNN Lenet-5
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import time


# 参数定义：
epoch_num = 20       # 迭代的次数
kernel_size = 5     # 卷积核的大小
hidden01_num = 10    # 通道一的个数
hidden02_num = 20   # 通道二的个数
Linear01_num = 4*4*hidden02_num   # 4是卷积后特征平面的纬度
Linear02_num = 120  # 全连接层一的输出个数
Linear03_num = 84   # 全连接层二的输出个数
Output_num = 10     # 输出层的个数
batch_size = 32

# 开始计时：
time_start = time.time()

# 固定随机数种子：
torch.manual_seed(100)

# 读取数据：
# 数据的大小为28*28
Dataset_train = torchvision.datasets.MNIST(root='../DataSet', download=True,
                                           train=True, transform=transforms.ToTensor())
Dataset_test = torchvision.datasets.MNIST(root='../DataSet',
                                          train=False, transform=transforms.ToTensor())
# 数据的个数：
Train_num = Dataset_train.data.shape[0]
Test_num = Dataset_test.data.shape[0]
# 输入特征的维度：
Input_size = Dataset_train.data.shape[1]

# 枚举器：
loader_train = torch.utils.data.DataLoader(dataset=Dataset_train, batch_size=batch_size, shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset=Dataset_test, batch_size=batch_size, shuffle=False)

# 输入特征，卷积参数，偏置项：
def MyConv(x, w, b):
    # x:批数，输入通道数，特征的维数，特征的维数
    # w:输出通道数，输入通道数，核的大小，核的大小：
    x_size = x.size(2)  # 特征平面的维度

    # 输出特征平面的维度：
    out_size = x_size - kernel_size + 1

    # 用unfold函数实现卷积运算：
    x_unfold = F.unfold(x, (kernel_size, kernel_size))    # [batch_size, kernel_size*kernel_size, out_size * out_size ]
    w_view = w.view(w.size(0), -1)  # 改变w的形状[batch_size, 后三维合成一维]
    out_mat = torch.matmul(w_view, x_unfold)  # 矩阵运算
    out = F.fold(out_mat, (out_size, out_size), (1, 1))   # [batch_size, output_num, out_size, out_size]
    x_conv = out + b   # 加上偏置项

    return x_conv


# 平均池化：
def MyAvePool(x):
    # x:批数，输入通道数，特征的纬度，特征的纬度
    x_batch = x.size(0)  # 批的个数
    x_dim = x.size(1)    # 输入的纬度
    x_size = x.size(2)  # 特征平面的纬度
    out_size = x_size // 2

    x_unfold = F.unfold(x, (2, 2), stride=2)   # 以步长2展开矩阵
    x_view = x_unfold.view(x_batch, x_dim, 4, out_size**2)
    x_mean = x_view.mean(2)  # 对第三列求均值
    x_out = x_mean.view(x_batch, x_dim, out_size, out_size)

    return x_out


# 模型定义：
class MyNet(nn.Module):   # 继承nn.Nodule
    def __init__(self):
        super(MyNet, self).__init__()
        # 输出通道数，输入通道数，核的大小
        # nn.Parameter()对需要求导的tensor操作, requires_grad默认为True
        self.W1 = nn.Parameter(torch.randn(hidden01_num, 1, kernel_size, kernel_size), requires_grad=True)
        # 用xavier对参数进行优化：
        nn.init.xavier_normal_(self.W1)
        self.b1 = nn.Parameter(torch.zeros(1, hidden01_num, 1, 1), requires_grad=True)
        # 第二层卷积：
        self.W2 = nn.Parameter(torch.randn(hidden02_num, hidden01_num, kernel_size, kernel_size), requires_grad=True)
        nn.init.xavier_normal_(self.W2)
        self.b2 = nn.Parameter(torch.zeros(1, hidden02_num, 1, 1), requires_grad=True)
        # 全连接层：
        self.L1 = nn.Parameter(torch.randn(Linear01_num, Linear02_num), requires_grad=True)    # 输入数据的纬度，输出数据的纬度
        self.Lb1 = nn.Parameter(torch.zeros(1, Linear02_num), requires_grad=True)
        self.L2 = nn.Parameter(torch.randn(Linear02_num, Linear03_num), requires_grad=True)
        self.Lb2 = nn.Parameter(torch.zeros(1, Linear03_num), requires_grad=True)
        self.L3 = nn.Parameter(torch.randn(Linear03_num, Output_num), requires_grad=True)
        self.Lb3 = nn.Parameter(torch.zeros(1, Output_num), requires_grad=True)

    def forward(self, x):
        # x是特征数据
        # 第一层卷积：
        x = MyConv(x, self.W1, self.b1)
        x = MyAvePool(x)
        # 激活函数：
        x = torch.sigmoid(x)

        # 第二层卷积：
        x = MyConv(x, self.W2, self.b2)
        x = MyAvePool(x)
        x = torch.sigmoid(x)

        # 全连接层：
        x = x.reshape(x.size(0), -1)  # 将各个特征平面展成一维向量

        x = torch.matmul(x, self.L1) + self.Lb1  # 矩阵运算
        x = torch.sigmoid(x)
        x = torch.matmul(x, self.L2) + self.Lb2
        x = torch.sigmoid(x)
        x = torch.matmul(x, self.L3) + self.Lb3
        # pytorch的cross_entropy包含softmax
        # x = torch.softmax(x, dim=1)  # 第一维是批的数量，第二位是输出的纬度，对第二位进行softmax()归一化

        return x


# 对象实例化：
model = MyNet()

# 设置优化器：
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

# 设置训练函数：
def train_and_loss(epoch):
    model.train()
    train_loss_all = []
    train_right_num = 0
    for batch_index, (x_train, y_train) in enumerate(loader_train):
        x_train, y_train = Variable(x_train), Variable(y_train)
        optimizer.zero_grad()
        output = model(x_train)
        # 计算Loss:
        loss = F.cross_entropy(output, y_train)
        train_loss_all.append(loss)
        # 预测正确的个数：
        max_index = output.max(dim=1).indices
        train_right_num += sum(max_index == y_train)
        # 反向传播：
        loss.backward()
        optimizer.step()
    # 训练集的平均Loss:
    train_loss = sum(train_loss_all) / len(train_loss_all)

    # 检验集的Loss：
    test_loss_all = []
    test_right_num = 0
    for batch_index, (x_test, y_test) in enumerate(loader_test):
        x_test, y_test = Variable(x_test), Variable(y_test)
        output_test = model(x_test)
        # 计算Loss:
        loss_test = F.cross_entropy(output_test, y_test)
        test_loss_all.append(loss_test)
        # 计算预测正确的个数：
        max_index = output_test.max(dim=1).indices
        test_right_num += sum(max_index == y_test)
    test_loss = sum(test_loss_all) / len(test_loss_all)

    return train_loss, test_loss, train_right_num, test_right_num


# 训练：
loss_all_train = []
loss_all_test = []
for epoch in range(epoch_num):
    [train_loss, test_loss, train_right_num, test_right_num] = train_and_loss(epoch)
    loss_all_train.append(train_loss)
    loss_all_test.append(test_loss)
    time_epoch = time.time()
    print('第{}轮，时间过了{}秒；'.format(epoch+1, time_epoch-time_start))
    # 正确率：
    train_ratio = int(train_right_num) / Train_num
    test_ratio = int(test_right_num) / Test_num
    print('训练集的正确率：{:.2f}%'.format(train_ratio*100))
    print('测试集的正确率：{:.2f}%'.format(test_ratio*100))


# 绘图：
x_label = np.linspace(1, epoch_num, epoch_num)
plt.figure()
plt.plot(x_label, loss_all_train, color='b', label='Train loss')
plt.plot(x_label, loss_all_test, color='r', label='Test loss')
plt.title('Loss via echo')
plt.xlabel('echo')
plt.ylabel('Loss')
plt.legend()
plt.show()


