import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
1、NiN块
'''
# 卷积层的输入和输出由四维张量组成，张量的每个轴分别对应样本、通道、高度和宽度。
# 另外，全连接层的输入和输出通常是分别对应于样本和特征的二维张量。

# NiN的想法是在每个像素位置（针对每个高度和宽度）应用一个全连接层。
# 如果我们将权重连接到每个空间位置，我们可以将其视为1X1卷积层，或作为在每个像素位置上独立作用的全连接层。
# 从另一个角度看，即将空间维度中的每个像素视为单个样本，将通道维度视为不同特征（feature）。
'''
即每个像素作为一个识别单元
每条通道对应一个分类
在每条通道上识别每个像素对应于该通道的概率(以值的形式最后softmax)。
(即图片的每个位置(像素)在1通道(第1种分类)的概率(以值的形式最后softmax)，然后对1通道上所有平均池化这种概率,得到该图片是种类1的概率)
(对2通道(第2种分类)同样进行这种操作)
(得出所有通道的值，softmax)
'''

# NiN块以一个普通卷积层开始，后面是两个1X1的卷积层。
# 这两个1X1卷积层充当带有ReLU激活函数的逐像素全连接层。
# 第一层的卷积窗口形状通常由用户设置。 随后的卷积窗口形状固定为1X1

import torch
from torch import nn
from d2l import torch as d2l


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    '''
    in_channels: 输入通道数
    out_channels: 输出通道数
    kernel_size: 第一个卷积层卷积核的大小
    strides:步长
    padding:padding
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
'''
第一个卷积层就定下输入输出的通道数
strides和padding都是用在第一个卷积层上的

后面两个1X1的卷积层默认步长和padding都是1，不改变通道数，其输入通道和输出通道数都是out_channels
两个1X1的卷积层后面都跟一个ReLU层
'''

'''
2、NiN模型
'''
# 最初的NiN网络是在AlexNet后不久提出的，显然从中得到了一些启示。
# NiN使用窗口形状为11X11、5X5和3X3的卷积层，输出通道数量与AlexNet中的相同。
# 每个NiN块后有一个最大池化层，池化窗口形状为3X3，步幅为2。

# NiN和AlexNet之间的一个显著区别是NiN完全取消了全连接层。
# 相反，NiN使用一个NiN块，其输出通道数等于标签类别的数量。
# 最后放一个全局平均池化层（global average pooling layer），生成一个对数几率（logits）。
'''
即生成一个值来softmax
'''
# NiN设计的一个优点是，它显著减少了模型所需参数的数量。



# 然而，在实践中，这种设计有时会增加训练模型的时间。


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())
'''
第一个nin块输入通道数为1，因为我们用的是灰度图
输出通道数是96(来自AlexNet的启发)
核大小为11步长为4，所以图片由224变成54
(224-11+0+4)/4=54.25,向下取整为54
'''
'''
最后一个NiN的块，把输出通道降到10，这个给到全局平均池化层的输入通道数，即类别数
'''
'''
注意最后的全局平均池化层的写法
nn.AdaptiveAvgPool2d((1, 1))
nn.AdaptiveAvgPool2d()接收一个元组作为输入，这个元组表示的是最终输出结果的高宽，这里1,1指的就是一个数字的意思。

最后的nn.Flatten()会保留最高一位批量大小，把后面的展平，所以最后出来的就是一个(批量大小，10)的二维数组，而高一位是批量大小。
即如下面测试的X的结果
tensor([[0.0000, 0.1694, 0.1275, 0.2393, 0.6344, 0.0000, 0.0000, 0.0000, 0.0457,
         0.2420]], grad_fn=<ReshapeAliasBackward0>)
批量大小为1，所以是[[  ]]
里面就是展平的10个通道的各个的值。
可以以此在train函数中进行softmax
'''
# 我们创建一个数据样本来查看每个块的输出形状。
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

'''
Sequential output shape:     torch.Size([1, 96, 54, 54])
MaxPool2d output shape:      torch.Size([1, 96, 26, 26])
Sequential output shape:     torch.Size([1, 256, 26, 26])
MaxPool2d output shape:      torch.Size([1, 256, 12, 12])
Sequential output shape:     torch.Size([1, 384, 12, 12])
MaxPool2d output shape:      torch.Size([1, 384, 5, 5])
Dropout output shape:        torch.Size([1, 384, 5, 5])
Sequential output shape:     torch.Size([1, 10, 5, 5])
AdaptiveAvgPool2d output shape:      torch.Size([1, 10, 1, 1])
Flatten output shape:        torch.Size([1, 10])
'''


'''
3、训练模型
'''
# 和以前一样，我们使用Fashion-MNIST来训练模型。训练NiN与训练AlexNet、VGG时相似。
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

'''
loss 0.363, train acc 0.865, test acc 0.879
3212.2 examples/sec on cuda:0
'''


'''
问题3：为什么最后只有一个全局池化层而不用softmax了

答：我们有softmax，这个过程是封装在我们training函数（train_ch6）中的,是在计算loss那里做的，没有放在网络上面，所以看不到
'''


'''
4、小结
'''
# NiN使用由一个卷积层和多个1X1卷积层组成的块。该块可以在卷积神经网络中使用，以允许更多的每像素非线性。

# NiN去除了容易造成过拟合的全连接层，将它们替换为全局平均池化层（即在所有位置上进行求和）。
# 该池化层通道数量为所需的输出数量（例如，Fashion-MNIST的输出为10）。
'''
池化层的输入通道数等于输出通道数
'''

# 移除全连接层可减少过拟合，同时显著减少NiN的参数。
