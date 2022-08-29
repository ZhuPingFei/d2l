import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

import torch
from torch import nn
from d2l import torch as d2l
'''
1、容量控制和预处理
'''
# AlexNet通过暂退法（ 4.6节）控制全连接层的模型复杂度，而LeNet只使用了权重衰减。
# 为了进一步扩充数据，AlexNet在训练时增加了大量的图像增强数据，如翻转、裁切和变色。
# 这使得模型更健壮，更大的样本量有效地减少了过拟合。

net = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    #
    #'''
    # 这里我们通道数设置为1，因为我们这里使用的是fashmnist，不是imagenet
    #'''
    #
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合

    #'''
    # 6400为展平后的数组大小,256(输出通道数)X5X5(池化完成后的矩阵大小，)
    # '''

    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # '''
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    # '''
    nn.Linear(4096, 10))


# 我们构造一个高度和宽度都为224的单通道数据，来观察每一层输出的形状。
# 它与AlexNet架构相匹配。
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
'''
注：下面是每层网络输出的四维变量，是具体值的格式的表述
'''
'''
卷积层创建的四维  和  输入输出卷积层的四维变量不同
前：(输入通道数，输出通道数，卷积核大小，是否有偏置单元)
后：从前往后为(批量大小、通道、高度、宽度)
见本章视频缺失章节和62图像卷积
'''
'''
Conv2d output shape:         torch.Size([1, 96, 54, 54])

注：这里54为
(输入224-核11+填充的上下和2+步幅4)/步幅4=54.75向下取整为54

ReLU output shape:   torch.Size([1, 96, 54, 54])
MaxPool2d output shape:      torch.Size([1, 96, 26, 26])

注：这里26为
(输入54-核3+步幅2)/步幅2=26.5向下取整为26

Conv2d output shape:         torch.Size([1, 256, 26, 26])

注：这里26为
输入26-核5+填充上下4+1=26

ReLU output shape:   torch.Size([1, 256, 26, 26])
MaxPool2d output shape:      torch.Size([1, 256, 12, 12])

注：这里12为
(输入26-核3+步幅2)/步幅2=12.5向下取整为12

Conv2d output shape:         torch.Size([1, 384, 12, 12])
ReLU output shape:   torch.Size([1, 384, 12, 12])

注：这里和下面的12为
输入12-核3+填充上下2+1=12

Conv2d output shape:         torch.Size([1, 384, 12, 12])
ReLU output shape:   torch.Size([1, 384, 12, 12])
Conv2d output shape:         torch.Size([1, 256, 12, 12])
ReLU output shape:   torch.Size([1, 256, 12, 12])
MaxPool2d output shape:      torch.Size([1, 256, 5, 5])

注：这里5为
(输入12-核3+步幅2)/步幅2=5.5向下取整为5
注：6400为展平后的数组大小,256(输出通道数)X5X5(池化完成后的矩阵大小)


Flatten output shape:        torch.Size([1, 6400])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 10])
'''
'''
此处，batchsize只有1，所以出来([1,10])数组，一个10个数字的数组(前面的1是批量大小)，前面多的一维1(批量大小的值)。
也就是最终出来的变量多个括号。
即例如[[0,0,0,0,0,0,0,0,1,0]]
这里夸张化了，softmax是使得这些加起来等于1


batchsize不为1时候，所以出来([X,10])数组,出来好几个10个数字的的数组，前面多的一维X就是批量大小的值
即例如[[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,1,0,0,0,0]]
这里夸张化了，softmax是使得这些加起来等于1
'''

'''
2、读取数据集
'''
# 尽管本文中AlexNet是在ImageNet上进行训练的，但我们在这里使用的是Fashion-MNIST数据集。
# 因为即使在现代GPU上，训练ImageNet模型，同时使其收敛可能需要数小时或数天的时间。
# 将AlexNet直接应用于Fashion-MNIST的一个问题是，Fashion-MNIST图像的分辨率（28X28像素）低于ImageNet图像。
# 为了解决这个问题，我们将它们增加到224X224
# （通常来讲这不是一个明智的做法，但我们在这里这样做是为了有效使用AlexNet架构）。
# 我们使用d2l.load_data_fashion_mnist函数中的resize参数执行此调整。
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
'''
此处的resize是李沐专门写在这个函数里的，不用深究这个resize
'''
'''
3、训练AlexNet
'''
# 现在，我们可以开始训练AlexNet了。
'''
# 与 6.6节中的LeNet相比，这里的主要变化是使用更小的学习速率训练.
AlexNet可以使用更小的学习率
'''

# 这是因为网络更深更广、图像分辨率更高，训练卷积神经网络就更昂贵。
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
loss 0.327, train acc 0.881, test acc 0.885
4149.6 examples/sec on cuda:0
'''
'''
LeNet跑时，能有近9万examples每秒
这里只有4000
慢了20倍。
但是LeNet的计算量比AlexNet少两百倍
因为LeNet不足以用尽GPU的核，所以每次GPU的计算是不充分调用的，并行度差
'''



