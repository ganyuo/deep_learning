#--coding:utf-8--
# 线性回归

import random
import torch
import matplotlib.pyplot as plt

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

# 定义线性回归模型
def linreg(x, w, b):
    return torch.matmul(x, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失函数。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features.shape, '\nlabels:', labels.shape)
    # plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    # plt.show()

    # batch_size = 10
    # for x, y in data_iter(batch_size, features, labels):
    #     print(x, '\n', y)
    #     break

    # 初始化模型参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 训练模型
    lr = 0.03
    num_epochs = 3
    batch_size = 10
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for x, y in data_iter(batch_size, features, labels):
            l = loss(net(x, w, b), y)  # l 的形状是(batch_size, 1)
            l.sum().backward()  # 小批量的损失对模型参数求梯度
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数

        # 查看训练结果的损失
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')

if __name__ == "__main__":
    main()
