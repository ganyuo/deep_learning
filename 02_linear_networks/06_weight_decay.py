#--coding:utf-8--
# 多层感知机，使用pytorch实现

import torch
import random
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

# 定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 定义线性回归模型
def linreg(x, w, b):
    return torch.matmul(x, w) + b

def main():

    # 生成训练集和验证集
    n_train, n_valid = 20, 100
    num_inputs = 200
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = synthetic_data(true_w, true_b, n_train)
    valid_data = synthetic_data(true_w, true_b, n_valid)

    # 初始化模型参数
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    net = linreg
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    lambd = 3  # 权重衰减的超参数
    # lambd = 0  # 权重衰减系数为0时，等价于不使用权重衰减

    for epoch in range(num_epochs):
        for x, y in data_iter(1, *train_data):
            l = loss(net(x, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()

        with torch.no_grad():
            train_l = loss(net(train_data[0], w, b), train_data[1])
            valid_l = loss(net(valid_data[0], w, b), valid_data[1])
            print(f'epoch {epoch + 1}, '
                  f'train loss {float(train_l):f}, '
                  f'valid loss {float(valid_l):f}')

if __name__ == "__main__":
    main()
