import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

'''
1、从零实现
'''
# 下面，我们从头开始实现一个具有张量的批量规范化层。
import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    '''
    X: 输入
    gamma: 可学习参数  BN处理后的方差
    beta: 可学习参数  BN处理后的均值
    moving_mean:全局的均值
    moving_var:全局的方差

    moving_mean和moving_var是整个数据集上的方差而不是小批量上的均值和方差
    是用来做inference用的

    eps:为了避免出0  (暂时不用想)  通常是一个固定值
    momentum:用来更新moving_mean和moving_var的东西，通常取固定的数字如0.9
    '''
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        '''
        用torch.is_grad_enabled来判断当前是否能计算梯度，如果是not，那么就说明是inference阶段
        此时计算X_hat
        '''
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        '''
        X减去当前数据集上全局的mean然后除以数据集上全局的var
        '''
        '''
        这里不可以用批量的均值，因为做预测时候经常会是一个样本进来，此时无法做批量的均值
        '''
    else:
        assert len(X.shape) in (2, 4)
        '''
        先用assert看一下X.shape的长度
        要么是2全连接层  (batch_size,feature)                (批量大小,特征)
        要么是4卷积层    (batch_size,channel,height,wide)    (批量大小,通道数,高,宽)
        
        否则报错不执行下列代码
        '''
        if len(X.shape) == 2:
            '''
            使用全连接层的情况，计算特征维上的均值和方差
            '''
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
            '''
            这里要求的是      
            每个样本中每一个特征的均值，然后拼出一个X形状的mean
            (这样才能让每个输入输出都在同一个大小范围不会梯度爆炸或者梯度消失)
            
            mean.shape = X[0].shape
            见75test文件
            
            mean的dim=0是指   
            按行求均值
            即对   每列求均值，在行上拼起来
            是计算特征维的均值
            是一个1Xn的行向量
            
            var同理
            
            见https://blog.csdn.net/Apikaqiu/article/details/104379960
            '''
        else:
            '''
            使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            
            即最后的结果留下三个通道的数据，每个通道的数据就是各个批量和高宽平均下来的值(一个数字)
            
            这里我们需要保持X的形状以便后面可以做   广播运算
            即每个批量的每个高宽位置的数字，可以以广播的形式来访问我们做出来的那个只有三个数字但是维度一样是4维的那个变量
            我们mean被广播放大到全部对应  通道  的所有位置都是那个数字，以此与X做计算
            '''
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            '''
            即在保证维度不变(4维)的情况下对通道维做平均
            (因为后面要计算，而且又是中间维度做整合，所以keepdim)
            
            出来的结果mean和var的形状
            1 X n X 1 X 1
            
            见75test的实验
            '''
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        '''
        这里就是BN计算后的X
        这个多个X一起满足正态分布
        
        之后计算的输出也就是Y也就是真正从BN层出去的feature X
        会用这个X_hat乘以可学习参数gamma加上可学习参数beta
        来学习改变BN层输出的分布来更适合模型
        '''
        '''
        前面是推理时候的计算
        用的是全局的mean和var
        这里是训练时候的计算
        用的是这次小批量的mean和var
        训练时候用小批量的mean和var也引入了随机抽样的噪声，使得模型泛化性更好
        '''

        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
        '''
        以momentum的比例把mean和moving_mean融合并更新。
        
        以0.9为例
        我们无法知道当前到底有多少个样本
        那么就把当前的均值乘以0.9在加上0.1乘以我当前小批量的均值
        这个叫指数移动加权平均
        这个只在训练的时候更新
        '''
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data


'''
最后返回   Y   和   全局mean和var的data(不要其的梯度)
'''


# 我们现在可以创建一个正确的BatchNorm层。
# 这个层将保持适当的参数：拉伸γ和偏移β,这两个参数将在训练过程中更新。
# 此外，我们的层将保存均值和方差的移动平均值，以便在模型预测期间随后使用。

# 撇开算法细节，注意我们实现层的基础设计模式。
# 通常情况下，我们用一个单独的函数定义其数学原理，比如说batch_norm。
# 然后，我们将此功能集成到一个自定义层中，
# 其代码主要处理数据移动到训练设备（如GPU）、分配和初始化任何必需的变量、跟踪移动平均线（此处为均值和方差）等问题。
# 为了方便起见，我们并不担心在这里自动推断输入形状，因此我们需要指定整个特征的数量。
# 不用担心，深度学习框架中的批量规范化API将为我们解决上述问题，我们稍后将展示这一点。
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            '''
            当dim大小为2，则其是全连接层
            那么形状就是(1, num_features)
            即(1代表对于多个批量做平均,num_features特征个数)
            '''
            shape = (1, num_features)
        else:
            '''
            否则就是卷积层
            形状就是(1, num_features, 1, 1)
            '''
            shape = (1, num_features, 1, 1)
        '''
        定义这些的原因是
        下面的四个参数都是长成这个形状
        以此来初始化这些参数
        gamma和var用1初始化
        beta和mean用0初始化
        gamma乘X_hat加beta，所以1、0
        (gamma相当于拟合的方差，beta相当于拟合的均值)
        moving_mean和moving_var用正态分布初始化，0、1
        '''
        '''
        其中，gamma和beta需要迭代更新
        moving_mean和moving_var不用迭代更新
        '''
        '''
        moving_mean和moving_var不是用梯度更新，所以不放在nn.parameter里
        '''
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        '''
        常见设定：
        eps = 1e-5
        momentum = 0.9
        
        在不同框架之间做迁移时，要注意这两个参数的大小
        '''
        return Y


'''
2、对LeNet使用批量归一化层
'''
# 为了更好理解如何应用BatchNorm，下面我们将其应用于LeNet模型
# 回想一下，批量规范化是在卷积层或全连接层之后、相应的激活函数之前应用的。
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
'''
第一个BN层
BatchNorm(6, num_dims=4)
上一层输出通道数是该层输入通道数，为6
因为上一层是卷积层，num_dims为4

后面的线性层后面的就比如BatchNorm(120, num_dims=2)

最后一个线性层不加，最后直接softmax的
'''
# 和以前一样，我们将在Fashion-MNIST数据集上训练网络。
# 这个代码与我们第一次训练LeNet时几乎完全相同，主要区别在于学习率大得多。
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
loss 0.268, train acc 0.900, test acc 0.831
38739.6 examples/sec on cuda:0
'''
# 让我们来看看从第一个批量规范化层中学到的拉伸参数gamma和偏移参数beta。
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
'''
(tensor([0.3362, 4.0349, 0.4496, 3.7056, 3.7774, 2.6762], device='cuda:0',
        grad_fn=<ReshapeAliasBackward0>),
 tensor([-0.5739,  4.1376,  0.5126,  0.3060, -2.5187,  0.3683], device='cuda:0',
        grad_fn=<ReshapeAliasBackward0>))
'''

'''
3、简明实现
'''
# 除了使用我们刚刚定义的BatchNorm，我们也可以直接使用深度学习框架中定义的BatchNorm。
# 该代码看起来几乎与我们上面的代码相同。
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
'''
此处不需要指定维度，里面自己会区分
'''
# 下面，我们使用相同超参数来训练模型。
# 请注意，通常高级API变体运行速度快得多，因为它的代码已编译为C++或CUDA，而我们的自定义代码由Python实现。
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
loss 0.269, train acc 0.901, test acc 0.853
64557.2 examples/sec on cuda:0
'''

'''
4、小结
'''
# 在模型训练过程中，批量归一化利用小批量的均值和标准差，不断调整神经网络的中间输出，使整个神经网络各层的中间输出值更加稳定。

# 批量归一化在全连接层和卷积层的使用略有不同。

# 批量规归一层和dropout层一样，在训练模式和预测模式下计算不同。

# 批量规范化有许多有益的副作用，主要是正则化。
# 另一方面，”减少内部协变量偏移“的原始动机似乎不是一个有效的解释。
'''
推论：BN是一种控制模型复杂度的方法，所以没必要和dropout一起混合使用
'''
