#--coding:utf-8--
# 多层感知机，使用pytorch实现

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

def accuracy(net, parameters, iter):
    """计算预测正确的概率"""
    accumlator = 0
    total_num = 0
    for x, y in iter:
        y_hat = net(x, parameters)
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        accumlator += float(cmp.type(y.dtype).sum())
        total_num += len(y)
    return accumlator / total_num if total_num > 0 else 0

def relu(x):
    """ReLU激活函数"""
    return torch.max(input=x, other=torch.tensor(0.0))

def net(x, parameters):
    """多层感知机模型"""
    w1, b1, w2, b2 = parameters
    num_inputs = w1.shape[0]
    x = x.reshape((-1, num_inputs))
    h1 = relu(torch.matmul(x, w1) + b1)
    return torch.matmul(h1, w2) + b2

def main():

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

    # 初始化模型参数
    num_inputs = 784
    num_hiddens = 256
    num_outputs = 10
    w1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True)
    b1 = torch.zeros(num_hiddens, requires_grad=True)
    w2 = torch.normal(0, 0.01, size=(num_hiddens, num_outputs), requires_grad=True)
    b2 = torch.zeros(num_outputs, requires_grad=True)
    params = [w1, b1, w2, b2]

    # 初始化损失函数
    loss = torch.nn.CrossEntropyLoss()

    # # 计算训练前的准确率
    # train_acc = accuracy(net, params, train_iter)
    # test_acc = accuracy(net, params, test_iter)
    # print(f'train acc {train_acc:f}, test acc {test_acc:f}')

    # 开始训练
    trainer = torch.optim.SGD(params, lr=0.1)
    epochs = 10
    for epoch in range(epochs):
        for x, y in train_iter:
            trainer.zero_grad()
            y_hat = net(x, params)
            l = loss(y_hat, y).mean()
            l.backward()
            trainer.step()
        train_acc = accuracy(net, params, train_iter)
        test_acc = accuracy(net, params, test_iter)
        print(f'epoch {epoch + 1}, loss {l:f}, train acc {train_acc:f}, test acc {test_acc:f}')

if __name__ == "__main__":
    main()
