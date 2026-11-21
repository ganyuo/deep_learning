#--coding:utf-8--
# 线性回归，使用pytorch的接口实现

import random
import torch
import torch.utils.data as data
from torch import nn

# 构造一个数据集
def synthetic_data(w, b, num_examples):
    # 生成 y = Xw + b + 噪声。
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    return x, y.reshape((-1, 1))

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

    # 初始化数据迭代器
    data_set = data.TensorDataset(features, labels)
    data_iter = data.DataLoader(data_set, batch_size=10, shuffle=True)
    # print(next(iter(data_iter)))

    # 初始化模型和模型参数
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 训练模型
    lr = 0.03
    batch_size = 10
    loss = nn.MSELoss() # 使用pytorch内置的均方误差损失函数
    trainer = torch.optim.SGD(net.parameters(), lr=lr) # 使用pytorch内置的随机梯度下降优化器

    num_epochs = 3
    for epoch in range(num_epochs):
        for x, y in data_iter:
            l = loss(net(x), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

if __name__ == "__main__":
    main()
