import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

'''
1、残差块
'''
# ResNet沿用了VGG完整的3X3卷积层设计。
# 残差块里首先有2个有相同输出通道数的3X3卷积层。
# 每个卷积层后接一个批量归一化层(BN层)和ReLU激活函数。
# 然后我们通过跨层数据通路，跳过这2个卷积运算，将输入直接加在最后的ReLU激活函数前。
# 这样的设计要求2个卷积层的输出与输入形状一样，从而使它们可以相加。
# 如果想改变通道数，就需要引入一个额外的1X1卷积层来将输入变换成需要的形状后再做相加运算。
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        '''
        input_channels: 输入通道数
        num_channels: 输出通道数
        use_1x1conv: 是否使用1X1卷积层
        strides: 第一个卷积层的步长，看是否把网络减半
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        '''
        第一个卷积层的步长为strides
        输入通道数为  总的输入通道数
        输出通道数 为 给定的输出通道数
        '''
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        '''
        第二个卷积层的步长为默认的1
        输入通道数和输出通道数都是  给定的输出通道数
        '''

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        '''
        用if判断是否使用了1X1卷积层
        来决定conv3是
        1、因为在ResNet块中改变了通道数，所以在加上x时候，要用1X1矩阵改变通道数。
        同样要用strides来改变网络形状，使得其与块的输出的高宽匹配。来更好的加起来。
        2、不做处理，直接作为一个连接上一层输出的通道
        '''

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        '''
        BN层有需要学习的参数
        所以要建立两个不同的BN层
        
        ReLU可以以层的方式加在里面，也可以像这里的F.relu(Y)来通过计算加入
        层的形式
        self.relu = nn.ReLU(inplace = True)
        
        inplace = True的意思
        正常情况下，ReLU新建一个output，把输入做max0（==跟0比大小，输出大的==）（即ReLU的操作）换过去

        inplace=true就是我不新建一个output了，改写input把值替换掉
        '''

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
        Y += X
        '''
        此处的逻辑
        判断self.conv3
        如果设定时要1X1卷积改变X大小
        则这里不是none
        则对X做操作后在函数外面与Y相加
        
        否则其是none
        直接让Y与X相加
        '''

        return F.relu(Y)


# 此代码生成两种类型的网络：
# 一种是当use_1x1conv=False时，应用ReLU非线性函数之前，将输入添加到输出。
# 另一种是当use_1x1conv=True时，添加通过1X1卷积调整通道和分辨率。

# 下面我们来查看输入和输出形状一致的情况。
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)
'''
torch.Size([4, 3, 6, 6])
'''

# 我们也可以在增加输出通道数的同时，减半输出的高和宽。
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)
'''
torch.Size([4, 6, 3, 3])
'''

'''
2、ResNet模型
'''
# ResNet的前两层跟之前介绍的GoogLeNet中的一样：
# 在输出通道数为64、步幅为2的7X7卷积层后，接步幅为2的3X3的最大汇聚层。
# 不同之处在于ResNet每个卷积层后增加了批量归一化层。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# GoogLeNet在后面接了4个由Inception块组成的模块。
# ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
# 第一个模块的通道数同输入通道数一致。
# 由于之前已经使用了步幅为2的最大池化层，所以无须减小高和宽。

# 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。

# 下面我们来实现这个模块。注意，我们对第一个模块做了特别处理。
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    '''
    这里是一个大的整合的ResNet块，可以认为是一个stage
    input_channels: 这个块的输入
    num_channels: 这个块的输出
    num_residuals: 总共有几个residual块

    first_block:  第一个块是否减半网络大小
    注：默认是False减半
    True是不减半，False是减半

    '''
    blk = []
    '''
    创建blk数组
    用来放入ResNet块中的每一个Residual小块
    '''
    for i in range(num_residuals):
        '''
        判断是否是第一个residual块以及    是否不进行通道减半
        如是
        则定义
        use_1x1conv=True, strides=2
        '''
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
'''
把blk数组作为函数的返回值
之后在nn.Sequential里解包组成网络
'''

# 接着在ResNet加入所有残差块，这里每个模块使用2个残差块。
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
'''
这里的2都表示一个ResNet块中有两个Residual块

first_block=True表示高宽不减半
除了第一个模块，其他都减半
第一个模块的通道数同输入通道数一致。
由于之前已经使用了步幅为2的最大池化层，所以无须减小  高和宽。
'''



# 最后，与GoogLeNet一样，在ResNet中加入全局平均池化层，以及全连接层输出。
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))
'''
按顺序排好stage
最后进行全局平均池化
然后展平全连接到对应的分类个数
'''
# 每个模块有4个卷积层（不包括恒等映射的1X1卷积层）。
# 加上第一个7X7卷积层和最后一个全连接层，共有18层。
# 因此，这种模型通常被称为ResNet-18。
# 通过配置不同的通道数和模块里的残差块数可以得到不同的ResNet模型，例如更深的含152层的ResNet-152。
# 虽然ResNet的主体架构跟GoogLeNet类似，但ResNet架构更简单，修改也更方便。
# 这些因素都导致了ResNet迅速被广泛使用


# 在训练ResNet之前，让我们观察一下ResNet中不同模块的输入形状是如何变化的。
# 在之前所有架构中，分辨率降低，通道数量增加，直到全局平均汇聚层聚集所有特征。
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

'''
Sequential output shape:     torch.Size([1, 64, 56, 56])
Sequential output shape:     torch.Size([1, 64, 56, 56])
Sequential output shape:     torch.Size([1, 128, 28, 28])
Sequential output shape:     torch.Size([1, 256, 14, 14])
Sequential output shape:     torch.Size([1, 512, 7, 7])
AdaptiveAvgPool2d output shape:      torch.Size([1, 512, 1, 1])
Flatten output shape:        torch.Size([1, 512])
Linear output shape:         torch.Size([1, 10])
'''

'''
3、训练模型
'''
# 同之前一样，我们在Fashion-MNIST数据集上训练ResNet。
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
loss 0.011, train acc 0.997, test acc 0.915
4701.1 examples/sec on cuda:0
'''

'''
4、小结
'''
# 学习嵌套函数（nested function）是训练神经网络的理想情况。
# 在深层神经网络中，学习另一层作为恒等映射（identity function）较容易（尽管这是一个极端情况）。
#
# 残差映射可以更容易地学习同一函数，例如将权重层中的参数近似为零。
#
# 利用残差块（residual blocks）可以训练出一个有效的深层神经网络：输入可以通过层间的残余连接更快地向前传播。
#
# 残差网络（ResNet）对随后的深层神经网络设计产生了深远影响。
