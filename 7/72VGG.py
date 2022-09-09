import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
1、VGG块
'''
# 经典卷积神经网络的基本组成部分是下面的这个序列：
# 1、带填充以保持分辨率的卷积层；
# 2、非线性激活函数，如ReLU；
# 3、池化层，如最大池化层。

# 而一个VGG块与之类似，由一系列卷积层组成，后面再加上用于空间下采样的最大汇聚层。
# 在最初的VGG论文中作者使用了
# 带有3X3卷积核、填充为1（保持高度和宽度）的卷积层，和带有2X2汇聚窗口、步幅为2（每个块后的分辨率减半）的最大池化层。
# 在下面的代码中，我们定义了一个名为vgg_block的函数来实现一个VGG块。
import torch
from torch import nn
from d2l import torch as d2l

'''
首先构建VGG块
'''
def vgg_block(num_convs, in_channels, out_channels):
    '''
    num_convs: 这次设立的VGG块有几个卷积层
    in_channels: 输入   这个VGG块时   有几个输入通道
    out_channels: VGG块中    每个卷积层有几个输出通道   以及  输出VGG块时的变量有几个输出通道
    '''
    layers = []
    for _ in range(num_convs):
        '''
        以VGG块有几个来做迭代
        '''
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        '''
        每次往layer数组中append上  卷积层 和 ReLU层
        卷积层的参数有固定kernel_size=3(3X3卷积核)和padding=1(保证网络大小不变)
        
        每层的   输入通道数
        (一开始为填入该VGG块的输入通道数。第一层卷积层结束，下一次调用输入通道数时就是下一次卷积层，会被赋值为该层输出通道数)   
        和  输出通道数(固定为设置的值，即概念中的m)
        '''
        in_channels = out_channels
        '''
        把输入通道数赋值为该层输出通道数供下一层来使用此作为输入通道数
        '''
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    '''
    加上最大池化层，stride=2，每次池化会让网络大小减半
    '''
    return nn.Sequential(*layers)

'''
*layers
意思为解包layer
表示取layer中的内容放到nn.Sequential中
即把我们设置的各个层每次append到layer这个数组中
然后按顺序取出来放到nn.Sequential里组成我们的VGG块

官方说法：把列表中的元素按顺序作为参数的输入函数
'''

'''
2、VGG网络
'''
# 与AlexNet、LeNet一样，VGG网络可以分为两部分：第一部分主要由卷积层和汇聚层组成，第二部分由全连接层组成。

# VGG神经网络连接的几个VGG块（在vgg_block函数中定义）。
# 其中有超参数变量conv_arch。该变量指定了每个VGG块里卷积层个数和输出通道数。
# 全连接模块则与AlexNet中的相同。

'''
原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。 


第一个模块有64个输出通道，每个后续模块将输出通道数量    翻倍    ，直到该数字达到512。

由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11。
'''
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
'''
VGG的架构，前面是每个VGG块中卷积层的个数，后面是VGG每个卷积层的输出通道数
'''
'''
为什么是5块？
因为图片是224X224，每经过一个VGG块的maxpooling就减半，经过5层减到7无法再除。
'''
# 下面的代码实现了VGG-11。可以通过在conv_arch上执行for循环来简单实现。
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    '''
    数据本身的输入通道数为1
    '''
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        '''
        for循环迭代访问conv_arch中的元组
        每个元组中代表   VGG中卷积层数  和  每个卷积层输出通道数
        赋值进去，并进行VGG块的初始化和连接(append到conv_blks里)
        '''
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        '''
        下一个块的输入通道数等于上一个块的输出通道数
        '''
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
'''
用conv_blks来不断按顺序append每个VGG块，最后解包
在nn.Sequential中，先是多个VGG块，后是3组全连接层(中间有ReLU和Dropout)
'''
net = vgg(conv_arch)

# 接下来，我们将构建一个高度和宽度为224的单通道数据样本，以观察每个层输出的形状。
'''
多通道一样，只不过要在入口对输入通道数做匹配，相当于只是改个数然后网络训练变得复杂点
'''
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

'''
Sequential output shape:     torch.Size([1, 64, 112, 112])
Sequential output shape:     torch.Size([1, 128, 56, 56])
Sequential output shape:     torch.Size([1, 256, 28, 28])
Sequential output shape:     torch.Size([1, 512, 14, 14])
Sequential output shape:     torch.Size([1, 512, 7, 7])
注：通道数翻倍，高宽减半(最后一层也可以翻倍成1024没关系，不过512很大一般不翻倍了)
这是一个经典的设计，之后会重复出现

Flatten output shape:        torch.Size([1, 25088])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([1, 4096])
Linear output shape:         torch.Size([1, 10])
'''
# 正如你所看到的，我们在每个块的高度和宽度减半，最终高度和宽度都为7。最后再展平表示，送入全连接层处理。

'''
3、训练模型
'''
# 由于VGG-11比AlexNet计算量更大，因此我们构建了一个通道数较少的网络，足够用于训练Fashion-MNIST数据集。
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
'''
pair是后面for循环的迭代，这里是一个语法糖
每次卷积层，输出通道数都除以4
这样可以减少些计算量
'''
net = vgg(small_conv_arch)
# 除了使用略高的学习率外，模型训练过程与AlexNet类似。
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
loss 0.177, train acc 0.934, test acc 0.911
2562.3 examples/sec on cuda:0
'''





