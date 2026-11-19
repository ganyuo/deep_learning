#--coding:utf-8--
# 自动求导

import torch

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

def main():

    # 自动求导
    x = torch.arange(4.0)
    x.requires_grad_(True)
    print(x.grad)
    y = 2 * torch.dot(x, x)
    print(y)
    y.backward()
    print(x.grad)
    print(x.grad == 4 * x)

    # 在默认情况下，PyTorch会累计梯度。我们需要手动清除它们
    x.grad.zero_()
    y = x.sum()
    y.backward()
    print(x.grad)

    # 非标量变量的反向传播
    x.grad.zero_()
    y = x * x
    y.sum().backward()
    print(x.grad)

    # 将某些计算结果从计算图中分离出来
    x.grad.zero_()
    y = x * x
    u = y.detach() # y.detach()返回一个常数张量，而不是x的函数。
    z = u * x
    print(u)
    print(z)
    z.sum().backward()
    print(x.grad)
    print(x.grad == u)
    x.grad.zero_()
    y.sum().backward()
    print(x.grad == 2 * x) # y还是是x的函数，所以梯度会被累加。

    # 复杂的控制流求导
    a = torch.randn(size=(), requires_grad=True)
    d = f(a)
    d.backward()
    print(a.grad == d / a)

if __name__ == "__main__":
    main()
