import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
axis是x，那么就相当于从左往右数x+1个维度消失

比如 254，axis = 0，变成54，两个54的数组按元素相加
254 axis = 1 变成24。两个分开，其中的54,5行案列相加变成4

就记住这个降维方法就会很好理解

axis可以为[0,1],此时就是消掉两个维度，即留下 4
'''


'''
如果是keepdimension的sum，那就是要把对应的维度变成1.

如254  axis =1 ，则为214
'''



'''
矩阵向量和矩阵矩阵点积思路见书本
'''

'''
矩阵向量。例 54矩阵 和  4向量
相当于5行4列，5行的每一行都是一个4向量，两个4向量点积，合为一个5向量

输入是横向量，输出是横向量

但是从数学角度理解是     54矩阵在前   乘了一个  4维的列向量，输出了一个5维的列向量

所以做运算的时候要会做。
'''


'''
矩阵矩阵
例 54矩阵  43矩阵
出来53矩阵

'''