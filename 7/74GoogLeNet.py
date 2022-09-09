import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
1、Inception块
'''
# Inception块由四条并行路径组成。 前三条路径使用窗口大小为1X1、3X3和5X5的卷积层，从不同空间大小中提取信息。
# 中间的两条路径在输入上执行1X1卷积，以减少通道数，从而降低模型的复杂性。
# 第四条路径使用3X3最大池化层，然后使用1X1卷积层来改变通道数。
# 这四条路径都使用合适的填充来使输入与输出的高和宽一致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出。

# 在Inception块中，通常调整的超参数是每层输出通道数。
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        '''
        重写原来的Inception块
        '''
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    '''
    分为4条路线，4条路线的超参数分别为c1，c2，c3，c4,
    其中23条路的超参数为元组，分别是卷积层的输出通道数
    '''
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
    '''
    拿到输入x
    丢到第一条路线，然后ReLU得到第一条的输出
    
    注：此处的函数ReLU计算，并非ReLU层。即接收参数输出结果的一个函数。
    这里是为了简单写就这样了
    
    '''
    '''
    最后torch.cat((p1, p2, p3, p4), dim=1)
    将结果在dim=1上合并，即批量大小dim=0通道数dim=1
    所以是在通道数这个维度上合并
    '''

# 那么为什么GoogLeNet这个网络如此有效呢？
# 首先我们考虑一下滤波器（filter）的组合，它们可以用各种滤波器尺寸探索图像，这意味着不同大小的滤波器可以有效地识别不同范围的图像细节。
# 同时，我们可以为不同的滤波器分配不同数量的参数。
'''
2、GoogLeNet
'''
# GoogLeNet一共使用9个Inception块和全局平均池化层的堆叠来生成其估计值。
# Inception块之间的最大池化层可降低维度(高宽)。
# 第一个模块类似于AlexNet和LeNet，Inception块的组合从VGG继承，全局平均池化层避免了在最后使用全连接层。

# 现在，我们逐一实现GoogLeNet的每个模块。第一个模块使用64个通道、7X7卷积层。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 第二个模块使用两个卷积层：第一个卷积层是64个通道、1X1卷积层；
# 第二个卷积层使用将通道数量增加三倍的3X3卷积层。 这对应于Inception块中的第二条路径。
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# 第三个模块串联两个完整的Inception块。
# 第一个Inception块的输出通道数为256。
# 第二个和第三个路径首先将输入通道的数量分别减少到96和16，然后连接第二个卷积层。

# 第二个Inception块的输出通道数增加到480。
# 第二条和第三条路径首先将输入通道的数量分别减少到128和32。
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
'''
Inception块的参数(在前面自定义的Inception块)：
(输入通道数，1X1卷积层的输出通道数，1X1和3X3卷积层的输出通道数的元组，1X1和5X5卷积层的输出通道数的元组)
'''

# 第四模块更加复杂， 它串联了5个Inception块，其输出通道数分别是512、512、512、528和832。
# 这些路径的通道数分配和第三模块中的类似，
# 首先是含3X3卷积层的第二条路径输出最多通道，其次是仅含1X1卷积层的第一条路径，之后是含5X5卷积层的第三条路径和含3X3最大汇聚层的第四条路径。
# 其中第二、第三条路径都会先按比例减小通道数。 这些比例在各个Inception块中都略有不同。
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 第五模块包含输出通道数为832和1024的两个Inception块。
# 其中每条路径通道数的分配思路和第三、第四模块中的一致，只是在具体数值上有所不同。
# 需要注意的是，第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均汇聚层，将每个通道的高和宽变成1。
# 最后我们将输出变成二维数组，再接上一个输出个数为标签类别数的全连接层。
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
'''
最后加一个全连接来映射
'''
# GoogLeNet模型的计算复杂，而且不如VGG那样便于修改通道数。
# 为了使Fashion-MNIST上的训练短小精悍，我们将输入的高和宽从224降到96，这简化了计算。
# 下面演示各个模块输出的形状变化。
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
'''
Sequential output shape:     torch.Size([1, 64, 24, 24])
Sequential output shape:     torch.Size([1, 192, 12, 12])
Sequential output shape:     torch.Size([1, 480, 6, 6])
Sequential output shape:     torch.Size([1, 832, 3, 3])
Sequential output shape:     torch.Size([1, 1024])
Linear output shape:         torch.Size([1, 10])
'''


'''
3、训练模型
'''
# 和以前一样，我们使用Fashion-MNIST数据集来训练我们的模型。在训练之前，我们将图片转换为96X96分辨率。
lr, num_epochs, batch_size = 0.03, 10, 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
loss 0.254, train acc 0.904, test acc 0.885
3570.5 examples/sec on cuda:0
'''

'''
4、小结
'''
# Inception块相当于一个有4条路径的子网络。
# 它通过不同窗口形状的卷积层和最大汇聚层来并行抽取信息，并使用1X1卷积层减少每像素级别上的通道维数从而降低模型复杂度。

# GoogLeNet将多个设计精细的Inception块与其他层（卷积层、全连接层）串联起来。
# 其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的。

# GoogLeNet和它的后继者们一度是ImageNet上最有效的模型之一：它以较低的计算复杂度提供了类似的测试精度。






































