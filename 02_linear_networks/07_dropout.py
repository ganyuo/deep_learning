#--coding:utf-8--
# 多层感知机，使用pytorch实现

import torch
import random
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn

# 构造一个数据集
def synthetic_data(w, b, num_examples):
    # 生成 y = Xw + b + 噪声。
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    return x, y.reshape((-1, 1))

# 数据集迭代器，按小批量读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def dropout(x, drop_prob):
    """对X的某些元素随机置0"""
    assert 0 <= drop_prob <= 1
    if drop_prob == 0:
        return x
    if drop_prob == 1:
        return torch.zeros_like(x)
    # torch.rand(x.shape) 生成一个与x形状相同的张量，元素在[0,1)之间均匀分布
    mask = (torch.rand(x.shape) > drop_prob).float() # 生成掩码
    return mask * x / (1.0 - drop_prob)

def accuracy(net, iter):
    """计算预测正确的概率"""
    accumlator = 0
    total_num = 0
    for x, y in iter:
        y_hat = net(x)
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        accumlator += float(cmp.type(y.dtype).sum())
        total_num += len(y)
    return accumlator / total_num if total_num > 0 else 0

# 定义模型
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens1 = num_hiddens1
        self.num_hiddens2 = num_hiddens2
        self.num_outputs = num_outputs
        self.linear1 = nn.Linear(num_inputs, num_hiddens1)
        self.linear2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2, num_outputs)
        self.dropout1_prob = 0  # 设置dropout1的概率
        self.dropout2_prob = 0  # 设置dropout2的概率

    def forward(self, x):
        x = x.reshape((-1, self.num_inputs))
        x = torch.relu(self.linear1(x))
        if self.training:  # 训练模式下使用dropout
            x = dropout(x, self.dropout1_prob)  # 在隐藏层使用dropout
        x = torch.relu(self.linear2(x))
        if self.training:  # 训练模式下使用dropout
            x = dropout(x, self.dropout2_prob)  # 在隐藏层使用dropout
        x = self.linear3(x)
        return x

def main():

    # 生成训练集和验证集
    # 读取数据集
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=trans, download=True)

    # 初始化数据迭代器
    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

    net = Net(784, 10, 256, 256)
    net.dropout1_prob = 0.2  # 设置dropout1的概率
    net.dropout2_prob = 0.5  # 设置dropout2的概率
    loss = nn.CrossEntropyLoss()
    num_epochs, lr = 10, 0.5
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 计算训练前的准确率
    train_acc = accuracy(net, train_iter)
    test_acc = accuracy(net, test_iter)
    print(f'train acc {train_acc:f}, test acc {test_acc:f}')

    for epoch in range(num_epochs):
        net.train()  # 在训练集上使用dropout
        for x, y in train_iter:
            l = loss(net(x), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()

        net.eval()  # 在验证集上不使用dropout
        train_acc = accuracy(net, train_iter)
        test_acc = accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, loss {l:f}, train acc {train_acc:f}, test acc {test_acc:f}')

if __name__ == "__main__":
    main()
