import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import os
import pandas as pd
# 首先创建一个人工数据集，并存储在CSV（逗号分隔值）文件
# csv是一个逗号分隔的文件，每行是一组数据，每个数据域用逗号分开

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


# 读取csv一般用pandas库
# 提供的函数read_csv来读取文件
data = pd.read_csv(data_file)
print(data)
'''
   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
'''
# 他会自动把第一行读为每个列的标题

# 把数据分成输入和输出特征
# 用iloc（index location）取列
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# fillna会把所有NaN（not a number）的值添上一个值，这里是除Na值以外的平均值
inputs = inputs.fillna(inputs.mean())
print(inputs)
'''
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
'''
# 对于inputs中的类别值或离散值，我们将“NaN”视为⼀个类别。
# 由于“巷⼦类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”
# pandas可以⾃动将此列转换为两列“Alley_Pave”和“Alley_nan”。\

# 通过dummies函数，dummy_na为是否把Nan视为一个类别
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
'''
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
'''

# 现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。
# 使用torch.tensor
X, Y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(Y)
'''
tensor([[3., 1., 0.],
        [2., 0., 1.],
        [4., 0., 1.],
        [3., 0., 1.]], dtype=torch.float64)
tensor([127500, 106000, 178100, 140000])
'''



