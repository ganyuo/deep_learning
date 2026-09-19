#--coding:utf-8--
# 数据基础操作

import torch

def main():
    x = torch.arange(12)
    print(x)
    print(x.shape)
    print(x.numel())

    x = x.reshape(3, 4)
    print(x)

    y = torch.zeros((2, 3, 4))
    print(y)

    y = torch.tensor([[0, 1, 2, 3],
                      [4, 5, 6, 7],
                      [8, 9, 10, 11]])
    print(y)

    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)

    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3],
                      [1, 2, 3, 4],
                      [4, 3, 2, 1]])
    print(torch.cat((x, y), dim=0))
    print(torch.cat((x, y), dim=1))

    print(x == y)
    print(x.sum())

    # 广播机制
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print(a + b)

    # 数据访问
    print(x[-1])
    print(x[1:3])

    # 数据写入
    x[1, 2] = 9
    print(x)
    x[0:2, :] = 12
    print(x)

    # 内存分配
    before = id(y)
    y = y + x
    print(id(y) == before)
    z = torch.zeros_like(y)
    print("id(z):", id(z))
    z[:] = x + y
    print("id(z):", id(z))
    before = id(x)
    x += y
    print(id(x) == before)

    # pytorch 与 NumPy 相互转换
    a = x.numpy()
    b = torch.from_numpy(a)
    print(type(a), type(b))

    # 转化为python标量
    a = torch.tensor([3.5])
    print(a, a.item(), float(a), int(a))

if __name__ == "__main__":
    main()