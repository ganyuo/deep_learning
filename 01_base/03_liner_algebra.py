#--coding:utf-8--
# 线性代数

import pandas as pd
import torch

def main():

    # 标量
    x = torch.tensor([3.0])
    y = torch.tensor([2.0])
    print(x + y)
    print(x * y)
    print(x / y)
    print(x ** y)

    # 向量
    x = torch.arange(4)
    print(x)
    print(x[3])
    print(len(x), x.shape)

    # 矩阵
    A = torch.arange(20).reshape(5, 4)
    print(A)
    print(A.T)

    # 对称矩阵
    B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
    print(B)
    print(B == B.T)

    # 多维张量
    x = torch.arange(24).reshape(2, 3, 4)
    print(x)

    # 张量运算
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    B = A.clone()
    print(A + B)
    print(A * B)
    a = 2
    print(a + A)
    print(a * A)

    # 张量属性
    x = torch.arange(4, dtype=torch.float32)
    print(x.shape)
    print(x.size())
    print(x.sum())
    print(x.mean())
    print(x.sum() / x.numel())
    print(A.sum(axis=1, keepdim=True)) # 按行求和，保持维度

    # 点积
    y = torch.ones(4, dtype=torch.float32)
    print(torch.dot(x, y))
    print(torch.matmul(A, x.reshape(4, 1)))

    # 范数
    u = torch.tensor([3.0, -4.0])
    print(torch.norm(u))
    print(torch.abs(u).sum()) # L1 范数
    u = torch.ones((4, 9), dtype=torch.float32)
    print(torch.norm(u, p=1)) # 矩阵 L1 范数
    print(torch.norm(u, p=2)) # 矩阵 L2 范数

if __name__ == "__main__":
    main()
