#--coding:utf-8--
# softmax回归

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表。"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def softmax(x):
    """定义softmax操作"""
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition  # 这里应用了广播机制

def net(x, param):
    """定义softmax回归模型"""
    w, b = param
    return softmax(torch.matmul(x.reshape((-1, w.shape[0])), w) + b)

def cross_entropy(y_hat, y):
    """定义交叉熵损失函数"""
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(net, param, iter):
    """计算预测正确的概率"""
    accumlator = 0
    total_num = 0
    for x, y in iter:
        y_hat = net(x, param)
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
    print(len(mnist_train), len(mnist_test))
    print(mnist_train[0][0].shape, mnist_train[0][1])

    # # 显示图像
    # x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    # # 测试加载数据时间
    # timer = d2l.Timer()
    # for x, y in train_iter:
    #     continue
    # print(f'{timer.stop():.2f} sec')

    # x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # x_prob = softmax(x)
    # print(x_prob, x_prob.sum(1))  # 每行的和为1

    # 初始化数据迭代器
    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

    # 初始化模型参数
    num_inputs = 784
    num_outputs = 10
    w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 计算训练前的准确率
    train_acc = accuracy(net, (w, b), train_iter)
    test_acc = accuracy(net, (w, b), test_iter)
    print(f'train acc {train_acc:f}, test acc {test_acc:f}')

    # 开始训练
    epochs, lr = 10, 0.1
    for epoch in range(epochs):
        for x, y in train_iter:
            y_hat = net(x, (w, b))
            l = cross_entropy(y_hat, y).mean()
            l.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
        train_acc = accuracy(net, (w, b), train_iter)
        test_acc = accuracy(net, (w, b), test_iter)
        print(f'epoch {epoch + 1}, loss {l:f}, train acc {train_acc:f}, test acc {test_acc:f}')

if __name__ == "__main__":
    main()
