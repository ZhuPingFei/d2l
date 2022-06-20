import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
视频缺失章节
4、1多层感知机
4、7前向传播反向传播计算图
4、9环境和分布偏移
'''




# @ 代表矩阵乘法

'''
flatten,展平成为一个二维。即前面是batchsize，后面是所有量乘起来合成一个量作为输入x。

如4 3 2
变
4  6
6就是x1x2到x6
(本来是x11x12x21x22x31x32)
'''

'''
丢弃法中
mask = (torch.rand(X.shape) > dropout).float()
张量和标量比较大小，大的出1小的出0，组成一个新的01张量
用.float使得其变成浮点数，因为之后函数返回值有除法
'''

'''
对于GPU，做乘法比选择效率更高
'''

'''
常用的trick：把隐藏层弄大，把dropout同时弄大。这样保持着差不多的过拟合可能，但是精度可能提高
'''