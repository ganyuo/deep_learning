---
title: 'PyTorch深度学习-基础操作'
published: 2026-11-20 17:30:00
category: 学习笔记
tags: [学习笔记,深度学习,PyTorch]
description: 学深度学习的过程中，记录下的学习笔记，大致分两块：“入门准备”（深度学习是什么、环境怎么搭）和“基础实践”（张量、数据预处理、线性代数、自动求导）。文中的代码都可以直接复制运行，按顺序敲一遍即可复现所有结果；个别片段会沿用上文定义过的变量，已在注释中标出。
author: ganyuo
pinned: false
draft: false
comment: true
---

> 本机环境：Miniconda 24.9.2 + 虚拟环境 `python39`（Python 3.9.20）+ `torch 2.8.0+cu128` + `pandas`
> 参考主线：《动手学深度学习》(d2l) 第 2 章「预备知识」

| 章节 | 内容 |
| --- | --- |
| 一、深度学习简介 | 深度学习的概念、训练流程、PyTorch 是什么 |
| 二、环境搭建 | 安装 Miniconda、创建虚拟环境、安装 PyTorch、验证 |
| 三、数据操作 | 张量怎么建、怎么算、怎么改、怎么和 NumPy 互转 |
| 四、数据预处理 | 用 pandas 处理缺失值和类别特征，再转成张量 |
| 五、线性代数 | 标量/向量/矩阵/张量的运算、求和、点积、范数 |
| 六、自动求导 | 计算图、`backward()`、梯度累加、`detach()` |
| 七 ~ 九 | API 速查、常见坑、小结 |

---

# 一、深度学习简介

## 1.1 深度学习是什么

深度学习（Deep Learning）是机器学习的一个分支：**用多层神经网络，从原始数据中自动学出「由简单到复杂」的特征表示**。

传统机器学习（如线性回归、SVM、决策树）通常要靠人先设计好特征（比如判断房价要手动挑「房间数」「地段」），模型只负责最后的映射；深度学习则把「提特征」这件事也交给模型，浅层学边缘/纹理/局部模式，深层把它们组合成物体、语义等高级概念，这叫**表示学习（representation learning）**。

一个网络的「深」，指的就是这种层层复合的结构。以最基础的多层感知机为例，一层的前向计算是

$$
\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})
$$

其中 $\mathbf{W}$ 是权重、$\mathbf{b}$ 是偏置、$\sigma$ 是非线性激活函数（如 ReLU）。多层堆叠起来，整个网络就是一个带参数的复合函数 $\hat{\mathbf{y}} = f(\mathbf{x}; \theta)$。

## 1.2 深度学习在解决什么问题

几乎所有深度学习任务都可以归纳成同一套框架：

1. **数据**：一堆输入 $\mathbf{x}_i$ 和对应的目标 $\mathbf{y}_i$；
2. **模型**：带参数 $\theta$ 的函数 $f(\mathbf{x};\theta)$；
3. **损失函数**：衡量预测与目标差多少，如均方误差 $L = \frac{1}{n}\sum_i \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|^2$、交叉熵；
4. **优化**：找让平均损失最小的参数

$$
\theta^{*} = \arg\min_{\theta} \frac{1}{n}\sum_{i=1}^{n} L\big(f(\mathbf{x}_i;\theta),\, \mathbf{y}_i\big)
$$

实际求解用**梯度下降**迭代：

$$
\theta \leftarrow \theta - \eta \,\nabla_{\theta} L
$$

其中 $\eta$ 是学习率。深度学习框架存在的意义，就是**自动算 $\nabla_\theta L$**（反向传播 + 自动求导）并高效执行矩阵运算——这正是第六章「自动求导」要动手验证的内容。

## 1.3 为什么是「现在」才火

三个条件同时成熟：**大数据**（互联网带来了海量标注数据）、**算力**（GPU 并行矩阵运算）、**算法改进**（ReLU 缓解梯度消失、Dropout 抑制过拟合、BatchNorm 稳定训练、残差连接让网络能真正做深）。

## 1.4 典型应用与局限

- 计算机视觉：图像分类、目标检测、语义分割、人脸识别；
- 自然语言处理：机器翻译、文本分类、大语言模型（GPT 类）；
- 语音：语音识别、语音合成；
- 其他：推荐系统、强化学习（游戏、机器人控制）、蛋白质结构预测等。

局限也要清楚：依赖大量数据与算力、训练成本高、可解释性差、对训练分布之外的数据（分布偏移）很脆弱、容易学到数据里的偏见。所以工程上数据质量和评测方式往往比「模型有多深」更关键。

## 1.5 PyTorch 简介

PyTorch 是由 Meta（原 Facebook）AI 团队于 2016 年开源、现由 Linux 基金会旗下 PyTorch Foundation 维护的深度学习框架。

它的几个关键特点：

- **动态图（define-by-run）**：计算图在代码运行时边执行边构建，写 `if`、`while`、变长循环都很自然，调试时可以像普通 Python 程序一样打断点（第六章最后的控制流求导例子就依赖这一点）；
- **Pythonic**：张量操作写法贴近 NumPy，学习成本低，和 Python 生态（NumPy、pandas、matplotlib）衔接顺畅；
- **自动求导**：`requires_grad` + `backward()` 自动完成反向传播；
- **生态完善**：`torchvision`（视觉）、`torchaudio`（音频）、`torchtext`（文本）、`torch.compile`（2.x 引入的图编译加速）、`DistributedDataParallel`（多卡训练）、ONNX 导出，以及 HuggingFace、d2l 等大量第三方库都优先支持 PyTorch；
- **研究与工业通吃**：论文复现基本首选 PyTorch，生产部署也有 TorchScript / `torch.compile` / ONNX 等路径。

和另外两个常见框架对比：TensorFlow 早期以静态图为主、部署生态强；JAX 走函数式 + 编译路线、偏研究；PyTorch 则在「灵活调试」和「性能」之间平衡得最好，是目前最主流的选择。

---

# 二、环境搭建

本节以 Linux 为例（Windows 见 2.2 末尾说明），目标是：**装 Miniconda → 建独立虚拟环境 → 装 PyTorch → 验证**。

## 2.1 为什么要用虚拟环境

不同项目依赖的 Python 版本、包版本往往互相冲突。虚拟环境（virtual environment）为每个项目提供一套独立的 Python 解释器和 `site-packages`，互不干扰；项目做废了直接删掉环境即可，不会污染系统 Python。

conda 除了管 Python 包，还能管非 Python 的二进制依赖（如 CUDA 运行库），所以深度学习场景一般首选 conda 系（Miniconda / Anaconda）。**Miniconda 只带 conda + Python，体积小**，需要的包自己装；Anaconda 预装了几百个包，体积大但省事。推荐 Miniconda。

## 2.2 安装 Miniconda

**Linux / macOS**：下载安装脚本并执行

```bash
# 下载（Linux x86_64；macOS 把文件名换成 Miniconda3-latest-MacOSX-arm64.sh）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 交互式安装（一路回车 + 输入 yes）
bash Miniconda3-latest-Linux-x86_64.sh
```

也可以静默安装到指定目录（本机就是这样装在 `~/programs/miniconda3`）：

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/programs/miniconda3

# 把 conda 写进 shell 配置，之后新开终端即可直接用 conda
~/programs/miniconda3/bin/conda init bash
source ~/.bashrc          # zsh 用户执行 source ~/.zshrc

conda --version           # 本机：conda 24.9.2
```

**Windows**：从 https://docs.conda.io/en/latest/miniconda.html 下载 `.exe` 安装包，双击按默认选项安装即可。安装时**不要**勾选「Add Miniconda3 to my PATH」（官方不建议），装完用开始菜单里的 **Anaconda Prompt** 操作，命令与下文一致。

如果官方源下载慢，可换清华镜像（可选）：

```bash
# pip 镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# conda 镜像：写入 ~/.condarc
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes
```

## 2.3 创建虚拟环境

```bash
# 创建名为 dl 的环境，指定 Python 3.11（版本按需改，3.9~3.12 都稳妥）
conda create -n dl python=3.11 -y

# 激活（每次新开终端都要激活一次）
conda activate dl

# 确认用的是环境里的解释器，而不是系统的
which python        # Windows: where python
python -V
```

本机实际用的是 Python 3.9 的环境，名字叫 `python39`：

```text
python39  *  /home/ganyuo/programs/miniconda3/envs/python39
```

常用 conda 命令：

| 目的 | 命令 |
| --- | --- |
| 查看所有环境 | `conda env list` |
| 激活 / 退出 | `conda activate dl` / `conda deactivate` |
| 查看已装包 | `conda list` |
| 安装包 | `conda install numpy` 或 `pip install numpy` |
| 删除环境 | `conda remove -n dl --all` |
| 导出 / 复现环境 | `conda env export > environment.yml` / `conda env create -f environment.yml` |
| 清理缓存 | `conda clean -a` |

## 2.4 安装 PyTorch

先确定要 **CPU 版**还是 **GPU 版**：

- 有 NVIDIA 显卡且装好了驱动（`nvidia-smi` 能正常输出）→ GPU 版；
- 没有显卡 / 只是学语法（本笔记的示例全在 CPU 上跑）→ CPU 版，体积小、装得快。

官方命令生成器：https://pytorch.org/get-started/locally/ 。下面是常用写法：

```bash
# CPU 版（pip，最省事）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU 版（pip，CUDA 12.8 轮子；CUDA 12.4 就把 cu128 换成 cu124）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# GPU 版（conda 写法，pytorch 官方渠道，版本更新通常比 pip 慢一些）
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

装完再补上示例要用的 pandas：

```bash
pip install pandas
```

注意：

- 一定要在**已激活的环境**里执行，否则会装到系统 Python 或 base 环境；
- 尽量别在同一个环境里反复 conda / pip 混装同一个包，容易版本打架；二进制依赖优先 conda，其余用 pip；
- GPU 版轮子体积很大（几个 GB），国内直连 `download.pytorch.org` 可能很慢，可以换用镜像或先下 wheel 再本地安装。

## 2.5 验证安装并运行示例

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

本机输出：

```text
2.8.0+cu128
12.8
False
```

解读：

- `2.8.0+cu128` 表示装的是**带 CUDA 12.8 的 GPU 版**；
- `torch.cuda.is_available() == False` 表示**当前用不了 GPU**——因为本机 `nvidia-smi` 报错（驱动没装好或没运行），不是 PyTorch 的问题。这时所有计算自动回退到 CPU（本机 16 核），学习阶段完全够用；
- 若 `torch.version.cuda` 打印 `None`，说明装成了纯 CPU 版。

最后做一次「冒烟测试」，确认张量运算和自动求导都正常：

```python
import torch

x = torch.arange(4.0, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)          # tensor([0., 2., 4., 6.])
```

能打印出 `tensor([0., 2., 4., 6.])`，就说明环境装好了。把这段存成 `test.py` 用 `python test.py` 运行，或者直接在终端里粘贴执行都可以。

用 VS Code 的话，按 `Ctrl+Shift+P` → `Python: Select Interpreter`，选中刚创建好的 conda 环境（如 `dl` 或 `python39`），之后点右上角的运行按钮即可。

## 2.6 常见问题

| 现象 | 原因 / 解决 |
| --- | --- |
| `conda: command not found` | 没执行 `conda init`，或执行后没重开终端；也可临时 `source ~/miniconda3/etc/profile.d/conda.sh` |
| 激活了环境但 `import torch` 失败 | 装到了别的环境里；`which python` 确认解释器路径，`conda list \| grep torch` 查包 |
| `torch.cuda.is_available()` 为 False | ① 装成了 CPU 版（`torch.version.cuda` 为 None）；② 驱动缺失/版本过低（`nvidia-smi` 报错）；③ 在无 GPU 的机器上 |
| 报 `Trying to backward through the graph a second time` | 同一个图重复 `backward()`，需要 `retain_graph=True` 或重新前向 |
| `ModuleNotFoundError: pandas` | 环境里没装，`pip install pandas` |
| 磁盘被 conda 占满 | `conda clean -a` 清缓存，`conda env list` 删掉不用的环境 |
| 装包时卡住 / 解析依赖很久 | 换镜像，或改用 `pip` 安装纯 Python 包 |

---

# 三、数据操作

张量（`torch.Tensor`）是 PyTorch 中存储和运算数据的基本单位，可以理解为「支持 GPU 与自动求导的多维数组」。

## 3.1 创建张量

```python
x = torch.arange(12)                 # [0, 1, ..., 11]，整数张量
x.shape                              # torch.Size([12])，形状
x.numel()                            # 12，元素总个数

x = x.reshape(3, 4)                  # 改成 3 行 4 列，仍是同一块数据

y = torch.zeros((2, 3, 4))           # 全 0，形状 (2, 3, 4)
y = torch.tensor([[0, 1, 2, 3],
                  [4, 5, 6, 7],
                  [8, 9, 10, 11]])   # 由 Python 列表构造
```

要点：

- `reshape` 只改变「视角」，不复制数据；元素总数必须保持不变。
- `shape` 返回 `torch.Size`（本质是 tuple），`numel()` 返回元素个数，二者相乘关系为
  $\prod_i \text{shape}_i = \text{numel}$。
- `torch.tensor(...)` 会根据 Python 字面量推断 dtype（有小数就是 `float32`）。

## 3.2 逐元素运算

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y   # 加、减、乘、除、幂，均为逐元素
```

这五种运算都是**按元素（element-wise）**进行的，形状相同即可直接算。此外还有 `torch.exp`、`torch.log`、`torch.abs` 等一元运算。

## 3.3 拼接、比较与求和

```python
x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3],
                  [1, 2, 3, 4],
                  [4, 3, 2, 1]])

torch.cat((x, y), dim=0)   # 沿第 0 维（行方向）拼接 → (6, 4)
torch.cat((x, y), dim=1)   # 沿第 1 维（列方向）拼接 → (3, 8)

x == y                     # 逐元素比较，返回布尔张量
x.sum()                    # 所有元素求和，返回 0 维张量
```

`dim=0` 是「上下叠」，`dim=1` 是「左右并」，拼接不改变其他维度大小。

## 3.4 广播机制

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a + b
# tensor([[0, 1],
#         [1, 2],
#         [2, 3]])
```

形状不同的张量相加时，PyTorch 会**广播**：先把大小为 1 的维度「复制扩展」到对方的大小。

规则（与 NumPy 一致）：从**最后一维**开始向前对齐，每一维要么相等，要么其中一个为 1，否则报错。

这里 $a$ 是 $3\times 1$，$b$ 是 $1\times 2$，结果为 $3\times 2$：

```text
a = [[0],      b = [[0, 1]]      结果第 (i, j) 项 = a[i] + b[j]
     [1],
     [2]]
```

## 3.5 索引与切片

```python
# 沿用 3.3 里的 x：(3, 4) 浮点张量
x[-1]        # 最后一行
x[1:3]       # 第 1、2 行（左闭右开）

x[1, 2] = 9      # 单个元素赋值
x[0:2, :] = 12   # 前两行整体赋值为 12
```

`x[1:3]` 这类切片返回的是**视图（view）**，修改它会影响原张量；若需要独立副本，用 `.clone()`。

## 3.6 内存与原地操作

这是新手最容易踩坑的地方：**Python 的 `y = y + x` 会新建一块内存，而 `y += x` 是原地修改**。

```python
# 沿用 3.3 里的 x 和 y（都是 3×4）
before = id(y)
y = y + x
print(id(y) == before)     # False，y 指向了新的张量

z = torch.zeros_like(y)    # 形状、dtype 与 y 相同
print("id(z):", id(z))
z[:] = x + y               # 用切片赋值把结果写回 z 原来的内存
print("id(z):", id(z))     # 地址不变

before = id(x)
x += y
print(id(x) == before)     # True，+= 是原地运算
```

结论：

- 重新赋值（`y = y + x`）→ 新内存，旧张量若无人引用会被回收；
- 切片赋值（`z[:] = ...`）、原地方法（`x += y`、`x.add_(y)`、`x.zero_()`）→ 复用内存。
- 原地操作在自动求导中要小心：它会破坏计算图所需的历史值，可能导致 `backward()` 报错。

## 3.7 与 NumPy 互转、转 Python 标量

```python
# 沿用上文的 x
a = x.numpy()          # Tensor → ndarray
b = torch.from_numpy(a)  # ndarray → Tensor

a = torch.tensor([3.5])
a.item()               # 3.5，取出 Python float
float(a), int(a)       # 3.5, 3（int 是截断，不是四舍五入）
```

重要区别：

- `torch.from_numpy(a)` 与 `a.numpy()` **共享底层内存**，改一个另一个也变（所以张量在 CPU 上才行）；
- `torch.tensor(a)` 则会**复制**数据。

`item()` 只适用于单元素张量；多元素张量要用 `tolist()`。

---

# 四、数据预处理

真实数据往往有缺失值和字符串类别特征，不能直接送进网络。流程是：

**读取 → 处理缺失值 → 编码类别特征 → 转成浮点张量**

## 4.1 读取数据

先准备一个小数据集：把下面内容存成 `house_tiny.csv`（和你的脚本放同一目录，只有 4 行，也可以手动在表格软件里录入）：

```text
NumRooms,Alley,Price
NA,Pave,127500
2,NA,106000
4,NA,178100
NA,NA,140000
```

然后用 pandas 读取：

```python
data = pd.read_csv("house_tiny.csv")
```

读进来的内容：

| NumRooms | Alley | Price |
| --- | --- | --- |
| NA | Pave | 127500 |
| 2 | NA | 106000 |
| 4 | NA | 178100 |
| NA | NA | 140000 |

pandas 会把 `NA` 自动识别成 `NaN`。

## 4.2 处理缺失值

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
```

- `iloc[:, 0:2]` 取前两列为特征 `inputs`，第 3 列为标签 `outputs`；
- `inputs.mean(numeric_only=True)` 只对**数值列**求均值，`Alley` 是字符串列不参与；
- `fillna(...)` 按列名对齐填充，于是 `NumRooms` 的缺失值被填成 $(2+4)/2 = 3$。

结果：

```text
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
```

注意 `Alley` 的缺失值仍然保留，留给下一步做独热编码。

## 4.3 类别特征编码（独热 / dummy）

```python
inputs = pd.get_dummies(inputs, dummy_na=True).astype(float)
```

`Alley` 只有 `Pave` 和缺失两类，`dummy_na=True` 把缺失也当成一个类别：

```text
   NumRooms  Alley_Pave  Alley_nan
0       3.0         1.0        0.0
1       2.0         0.0        1.0
2       4.0         0.0        1.0
3       3.0         0.0        1.0
```

`.astype(float)` 把布尔列转成浮点，保证整张表类型统一。

## 4.4 转为张量

```python
x = torch.tensor(inputs.values, dtype=torch.float32)
y = torch.tensor(outputs.values, dtype=torch.float32)
```

```text
x = tensor([[3., 1., 0.],
            [2., 0., 1.],
            [4., 0., 1.],
            [3., 0., 1.]])
y = tensor([127500., 106000., 178100., 140000.])
```

要点：

- 神经网络参数默认是 `float32`，所以特征、标签都显式指定 `dtype=torch.float32`；
- `.values` 拿到 ndarray 后交给 `torch.tensor` 会复制一份，与 DataFrame 解耦，避免后续改动互相影响。

---

# 五、线性代数

## 5.1 标量、向量、矩阵、张量

```python
x = torch.tensor([3.0])   # 标量（这里用长度为 1 的张量表示）
x = torch.arange(4)       # 向量，shape = (4,)，x[3] 取第 4 个元素
A = torch.arange(20).reshape(5, 4)   # 矩阵，5 行 4 列
A.T                                  # 转置 → (4, 5)
x = torch.arange(24).reshape(2, 3, 4)  # 三维张量
```

维度直觉：标量 0 维 → 向量 1 维 → 矩阵 2 维 → 张量 $n$ 维。`len(x)` 只返回第 0 维长度，看全貌请用 `x.shape`。

对称矩阵满足 $B = B^\top$：

```python
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B == B.T    # 全 True
```

## 5.2 张量运算与克隆

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()      # 深拷贝，独立内存
A + B              # 逐元素加
A * B              # 哈达玛积（逐元素乘），不是矩阵乘法！
a = 2
a + A, a * A       # 标量与张量运算，逐元素广播
```

> 注意：`A * B` 是**按元素乘**，矩阵乘法要用 `torch.matmul` / `@`。

## 5.3 求和、均值与 `keepdim`

```python
x = torch.arange(4, dtype=torch.float32)   # [0, 1, 2, 3]
x.sum()                # 6
x.mean()               # 1.5
x.sum() / x.numel()    # 1.5，与 mean 等价

A.sum(axis=1, keepdim=True)   # 每行求和，shape (5, 1)
```

- `mean()` 要求浮点 dtype，整数张量直接调用会报错，所以这里用 `dtype=torch.float32`；
- `axis` 与 `dim` 同义，指定沿哪个维度归约；
- `keepdim=True` 保留被归约的维度（大小变为 1），这样结果还能和原张量做广播运算。例如行求和后想「每行除以自己的和」，必须保持 $(5,1)$ 形状才能广播。

## 5.4 点积与矩阵-向量积

```python
y = torch.ones(4, dtype=torch.float32)
torch.dot(x, y)                        # 6，一维向量内积
torch.matmul(A, x.reshape(4, 1))       # (5, 1) 矩阵乘向量
```

数学上：

```text
torch.dot(x, y) = Σ_i x_i·y_i
torch.matmul(A, v) 的第 i 项 = Σ_j A[i, j]·v_j
```

`torch.dot` 只接受一维张量；矩阵乘法用 `torch.matmul`、`torch.mm`（二维）或运算符 `@`。

## 5.5 范数

范数用来衡量向量的「长度」或张量的「大小」，也是正则化项的基础。

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)              # 5.0，默认 L2 范数
torch.abs(u).sum()         # 7.0，L1 范数

u = torch.ones((4, 9), dtype=torch.float32)
torch.norm(u, p=1)         # 36.0 = 4*9，所有元素绝对值之和
torch.norm(u, p=2)         # 6.0  = sqrt(36)，Frobenius 范数
```

对应公式：

$$
\|\mathbf{x}\|_1 = \sum_{i} |x_i|, \qquad
\|\mathbf{x}\|_2 = \sqrt{\sum_{i} x_i^2}
$$

$$
\|X\|_F = \sqrt{\sum_{i}\sum_{j} x_{ij}^2}
$$

**易错点**：`torch.norm(矩阵, p=1)` 并不等于「列和的最大值」，而是把矩阵**展平后**当作向量求 L1 范数（本例 $4\times 9$ 全 1 矩阵得 36）。若需要真正的矩阵范数，用：

```python
torch.linalg.matrix_norm(A, ord=1)   # 最大列和
torch.linalg.matrix_norm(A, ord='fro')  # Frobenius
torch.linalg.norm(A, dim=0)          # 按维度求范数
```

L2 范数（Frobenius 范数）在实践中常用于权重衰减，防止参数过大。

---

# 六、自动求导

PyTorch 通过**计算图**自动计算梯度，这是训练神经网络的核心机制。

## 6.1 基本流程：`requires_grad` → 前向 → `backward`

```python
x = torch.arange(4.0)
x.requires_grad_(True)     # 声明需要求导（也可在创建时 requires_grad=True）
print(x.grad)              # None，还没反向传播过

y = 2 * torch.dot(x, x)    # 前向计算，同时构建计算图
y.backward()               # 反向传播，梯度写入 x.grad

print(x.grad)              # tensor([0., 4., 8., 12.])
print(x.grad == 4 * x)     # 全 True
```

推导：$y = 2\sum_i x_i^2$，所以

$$
\frac{\partial y}{\partial x_i} = 4x_i
$$

`x.grad` 是**累加**的结果，所以用之前一定要清零。

## 6.2 梯度会累加，记得 `zero_()`

```python
x.grad.zero_()          # 手动清零
y = x.sum()
y.backward()
print(x.grad)           # tensor([1., 1., 1., 1.])
```

若不清零，第二次 `backward()` 的结果会叠加在旧梯度上。训练循环里标准写法是：

```python
# 伪代码：假设已经定义了 model、optimizer 和 loss
optimizer.zero_grad()   # 或 model.zero_grad()
loss.backward()
optimizer.step()
```

## 6.3 非标量输出：为什么要 `.sum()`

```python
x.grad.zero_()
y = x * x               # y 是向量
y.sum().backward()      # 先聚合成标量，再反传
print(x.grad)           # tensor([0., 2., 4., 6.])
```

`backward()` 需要一个标量来定义「对谁的梯度」。对向量调用时，PyTorch 默认等价于先乘一个全 1 向量再反传（即隐式梯度为 $\mathbf{1}$），但显式写 `.sum().backward()` 更清晰、不易出错。

## 6.4 `detach()`：把子表达式从计算图中摘出来

```python
x.grad.zero_()
y = x * x
u = y.detach()      # u 与 y 数值相同，但被当作常数，不再有梯度
z = u * x
z.sum().backward()
print(x.grad)       # tensor([0., 1., 4., 9.])，即 u 的值
print(x.grad == u)  # 全 True

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)   # 全 True
```

含义：

- $z = u \cdot x$，其中 $u$ 被视为常数，故 $\partial z/\partial x = u$；
- 而 `y` 仍然记录着「$y = x^2$」的图，所以对 `y.sum()` 反传仍得到 $2x$。

`detach()` 的典型用途：

1. 把某个张量当**常量**使用（如目标值、固定权重）；
2. 截断梯度（如 GAN 训练判别器时不让梯度传回生成器）；
3. 转成 NumPy / 打印时避免带着计算图（或用 `.detach().numpy()`）。

## 6.5 控制流求导

```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)
```

这段代码说明：**动态计算图会记录实际执行过的路径**。最终 $d = k \cdot a$（$k$ 是某个由 `while`/`if` 决定的常数），因此

$$
\frac{\partial d}{\partial a} = k = \frac{d}{a}
$$

无论中间循环了多少次、走了哪个分支，PyTorch 都能正确求导——这正是「动态图」相比静态图灵活的地方。

> **小坑**：这里用 `==` 比较浮点数，偶尔会因为舍入误差输出 `False`（尤其 `c = 100 * b` 分支里乘以 100 不是 2 的幂）。实际项目中应使用
> ```python
> torch.allclose(a.grad, d / a)
> ```
> 或 `torch.isclose(...).all()`。

---

# 七、API 速查表

| 需求 | API |
| --- | --- |
| 创建 | `torch.tensor`、`torch.arange`、`torch.zeros`、`torch.ones`、`torch.zeros_like`、`torch.randn` |
| 形状 | `x.shape`、`x.size()`、`x.numel()`、`x.reshape`、`x.T` |
| 拼接 | `torch.cat(tensors, dim)`、`torch.stack` |
| 逐元素运算 | `+ - * / **`、`torch.exp`、`torch.log`、`torch.abs` |
| 归约 | `x.sum(dim, keepdim)`、`x.mean`、`x.max`、`x.argmax` |
| 线性代数 | `torch.dot`、`torch.matmul` / `@`、`torch.norm`、`torch.linalg.matrix_norm` |
| 内存 | `x.clone()`、`x[:] = ...`、`x.add_()`、`id(x)` |
| NumPy | `x.numpy()`、`torch.from_numpy(a)`、`x.item()` |
| 预处理 | `pd.read_csv`、`df.fillna`、`df.mean(numeric_only=True)`、`pd.get_dummies(dummy_na=True)` |
| 自动求导 | `requires_grad_`、`backward()`、`x.grad`、`x.grad.zero_()`、`x.detach()` |

---

# 八、常见坑总结

1. **`A * B` 不是矩阵乘法**，是逐元素乘；矩阵乘用 `torch.matmul` 或 `@`。
2. **`y = y + x` 与 `y += x` 内存行为不同**，后者原地修改，`id` 不变。
3. **切片是视图**，改切片会改原张量；需要独立副本请 `.clone()`。
4. **`from_numpy` / `numpy()` 共享内存**，`torch.tensor(ndarray)` 才复制。
5. **`mean()` 要求浮点张量**，整数张量会报错，记得 `dtype=torch.float32`。
6. **梯度是累加的**，每次 `backward()` 前要 `zero_()`。
7. **非标量必须转成标量再 `backward()`**，通常用 `.sum()`。
8. **`keepdim=True` 是广播的前提**，按行归一化等操作别忘了。
9. **`torch.norm(矩阵, p=1)` 是展平后的 L1 范数**，矩阵范数请用 `torch.linalg`。
10. **浮点数不要用 `==` 比较**，用 `torch.allclose`。
11. **`int(tensor)` 是截断**，`3.9` 会变成 `3`。
12. **原地操作会破坏计算图**，在需要反传的张量上慎用 `+=`、`zero_()` 之外的原地方法。

---

# 九、小结

从搭环境到基础实践，整条学习路径是：

```text
环境（Miniconda + 虚拟环境 + PyTorch）
        ↓
张量（数据载体）
  ├── 创建 / 形状 / 索引 / 内存管理 / 与 NumPy 互转
  ├── 运算：逐元素、拼接、广播、归约
  └── 线性代数：转置、点积、矩阵乘、范数
        ↓
数据预处理（pandas）
  ├── 缺失值填充（均值）
  ├── 类别特征独热编码
  └── 转成 float32 张量
        ↓
自动求导（计算图）
  ├── requires_grad → 前向建图 → backward 求梯度
  ├── 梯度累加 → zero_()
  ├── 非标量 → sum() 聚合
  └── detach() 截断梯度 / 当常量用
        ↓
下一步：线性回归 → 多层感知机 → 训练循环
```

一句话概括：**张量负责存和算，pandas 负责把脏数据洗成张量，自动求导负责把「损失对参数的导数」算出来**——三者合起来就是神经网络的训练闭环。
