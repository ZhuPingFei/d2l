import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
X=torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
'''
X是(3,4)的一个tensor
batchsize 3
feature 4
'''
mean1 = X.mean(dim=0)
mean2 = X.mean(dim=1)
print(mean1)
print(mean2)
'''
tensor([2.3333, 2.0000, 3.0000, 2.6667])
tensor([2.5000, 2.5000, 2.5000])
'''
print(X[0].shape)
print(mean1.shape)
'''
torch.Size([4])
torch.Size([4])
'''

'''
卷积层
设高宽为2，通道数为3，批量为4
'''
X=torch.tensor( [   [  [[2.0, 1], [1, 2]] , [[4, 2], [4, 2]]  ,  [[8, 4], [8, 4]] ] ,
                    [  [[2.0, 1], [1, 2]] , [[4, 2], [4, 2]]  ,  [[8, 4], [8, 4]] ]  ,
                    [  [[2.0, 1], [1, 2]] , [[4, 2], [4, 2]]  ,  [[8, 4], [8, 4]] ]  ,
                    [[[2.0, 1], [1, 2]],     [[4, 2], [4, 2]],   [[30, 30], [30, 30]]]
                 ])

print(X.shape)
'''
torch.Size([4, 3, 2, 2])
'''
mean1 = X.mean(dim=(0, 2, 3))
mean2 = X.mean(dim=(0, 2, 3), keepdim=True)
print(mean1)
print(mean2)
'''
tensor([ 1.5000,  3.0000, 12.0000])
tensor([[[[ 1.5000]],

         [[ 3.0000]],

         [[12.0000]]]])
'''
'''
从上面结果可以看出
mean2 = X.mean(dim=(0, 2, 3), keepdim=True)
先把dim3平均，dim3上一个数
此时在dim2上就变成两个数，然后在平均
dim2上变成一个数

dim3，4上此时就是    横着3个数，竖着四行

再在dim4上平均
按列，竖着的都合并

最后变成3个数
形状torch.Size([3])

如果keepdim那么形状就torch.Size([1, 3, 1, 1])
从抽象理解就是仅保留我们剩下的做平均的维度，其余为1

我们要在哪个维度做平均，我们就在其余维度mean(dim=)

'''



print(mean1.shape)
print(mean2.shape)
print(X.shape)
'''
torch.Size([3])
torch.Size([1, 3, 1, 1])
torch.Size([4, 3, 2, 2])
'''
'''

'''