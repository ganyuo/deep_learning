#--coding:utf-8--
# softmax回归，使用pytorch实现

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

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

    # 初始化模型、模型参数、损失函数
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
    for param in net.parameters():
        torch.nn.init.normal_(param, mean=0, std=0.01)
    loss = torch.nn.CrossEntropyLoss(reduction='none')

    # 计算训练前的准确率
    train_acc = accuracy(net, train_iter)
    test_acc = accuracy(net, test_iter)
    print(f'train acc {train_acc:f}, test acc {test_acc:f}')

    # 开始训练
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    epochs = 10
    for epoch in range(epochs):
        for x, y in train_iter:
            trainer.zero_grad()
            y_hat = net(x)
            l = loss(y_hat, y).mean()
            l.backward()
            trainer.step()
        train_acc = accuracy(net, train_iter)
        test_acc = accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, loss {l:f}, train acc {train_acc:f}, test acc {test_acc:f}')

if __name__ == "__main__":
    main()
