#--coding:utf-8--
# 数据预处理

import pandas as pd
import torch

def main():

    # 读取数据
    data = pd.read_csv("01_base/data/house_tiny.csv")
    print(data)

    # 数据插值
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean(numeric_only=True))
    print(inputs)
    print(outputs)

    # 数据转换
    inputs = pd.get_dummies(inputs, dummy_na=True).astype(float)
    print(inputs)

    # 转换为 PyTorch 张量
    x = torch.tensor(inputs.values, dtype=torch.float32)
    y = torch.tensor(outputs.values, dtype=torch.float32)
    print(x)
    print(y)

if __name__ == "__main__":
    main()