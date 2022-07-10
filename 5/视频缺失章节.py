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
5、3延后初始化。（只有MXnet和tensorflow，搞不明白这个是干嘛用的，好像完全不影响）
'''
# 我们一直在通过net(X)调用我们的模型来获得模型的输出。
# 这实际上是net.__call__(X)的简写。(语法糖)
# 这个__call__函数调用module函数中的forward函数，所以所有其子类的前向传播就是重现forward函数
######################################################################
# 下面的很重要，细读
'''
Sequential中加层
1、弄一个class，然后在Sequential中是初始化一个新的，加两次不会绑定（51）
这样是说明神经网络构建的灵活性


2、初始化一个层后赋值给一个变量，多次加到层里面，绑定（52）相当于share权重
在sequential中套一个灵活加入的层不一样
那个是class，相当于每次重新初始化
而这个是把一个确切的层的实例加入
'''
# 重要结束

'''
52中nn.init中有很多函数
比如.normal_

XXXX_在python中的意思是会对读到的输入进行改变
'''

'''
net来apply这个函数，即对所有   层（module）  进行该函数
假设是嵌套，那么就会嵌套遍历
即会递归调用，直到所有 层 都进行该函数

也可以net[0].apply(xavier)只对一层进行
'''


