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
from torch.nn import functional as F

# 下面的代码生成一个网络
# 其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层
# 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。
'''
nn.Sequential是一个特殊的module类，其定义   块
'''
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
print(net(X))
'''
tensor([[ 0.0571, -0.1483,  0.0634,  0.2324,  0.0511, -0.0160, -0.3767,  0.0174,
          0.1023,  0.1853],
        [ 0.1198, -0.2116,  0.0051,  0.2289,  0.1122,  0.0550, -0.3138, -0.0873,
         -0.2142,  0.0671]], grad_fn=<AddmmBackward0>)
'''
# 在这个例子中，我们通过实例化nn.Sequential来构建我们的模型
# 层的执行顺序是作为参数传递的。
# nn.Sequential定义了一种特殊的Module
# 即在PyTorch中表示         一个块        的类， 它维护了一个由Module组成的有序列表。
# 注意，两个全连接层都是Linear类的实例， Linear类本身就是Module的子类。


# 我们一直在通过net(X)调用我们的模型来获得模型的输出。
# 这实际上是net.__call__(X)的简写。
# 这个前向传播函数非常简单： 它将列表中的每个块连接在一起，将每个块的输出作为下一个块的输入。

'''
1、自定义块
'''
# 在实现我们自定义块之前，我们简要总结一下每个块必须提供的基本功能：
# 1、将输入数据作为其前向传播函数的参数。
# 2、通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。
# 例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
# 3、计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
# 4、存储和访问前向传播计算所需的参数。
# 5、根据需要初始化模型参数。

# 在下面的代码片段中，我们从零开始编写一个块。
# 它包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。
# 注意，下面的MLP类继承了表示块的类。
# 我们的实现只需要提供我们自己的构造函数（Python中的__init__函数）和前向传播函数。
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        '''
        调用MLP的父类Module的构造函数来执行必要的初始化。
        即在初始化神经网络的内部参数
        即weight、bias等等
        '''
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        '''
        把nn.functional记为F，里面有很多函数。
        此处F.relu与nn.ReLU()不一样
        前者为一个函数后者为一个神经网络的激活层
        此处是用其作为一个函数
        '''
        return self.out(F.relu(self.hidden(X)))

'''
注意，除非我们实现一个新的运算符， 否则我们不必担心反向传播函数或参数初始化， 系统将自动生成这些。
'''
net = MLP()
print(net(X))
'''
tensor([[-1.3925e-02,  2.2925e-01, -9.7376e-02,  1.2108e-01, -9.3669e-02,
         -5.3436e-05,  1.3887e-01,  7.2933e-02, -2.9266e-02, -5.0701e-03],
        [-1.2001e-01,  1.6734e-01,  5.3979e-02,  1.3259e-01, -1.6270e-01,
         -5.2622e-02,  1.0470e-02, -7.2272e-02, -5.0206e-02,  2.6640e-02]],
       grad_fn=<AddmmBackward0>)
'''
'''
块的一个主要优点是它的多功能性。
我们可以子类化块以创建层（如全连接层的类）、 整个模型（如上面的MLP类）或具有中等复杂度的各种组件。 
我们在接下来的章节中充分利用了这种多功能性， 比如在处理卷积神经网络时。
'''

'''
2、顺序块
'''
# 现在我们可以更仔细地看看Sequential类是如何工作的，
# 回想一下Sequential的设计是为了把其他模块串起来。
# 为了构建我们自己的简化的MySequential， 我们只需要定义两个关键函数：
# 1、一种将块逐个追加到列表中的函数。
# 2、一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。
# 下面的MySequential类提供了与默认Sequential类相同的功能。
class MySequential(nn.Module):
    def __init__(self, *args):
        '''
        *args
        指的是list of input arguments
        *args是收集参数，相当于把多个参数打包成一个来传入
        '''
        super().__init__()
        for idx, module in enumerate(args):
            '''
            for循环args里面所有的层
            enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
            同时列出数据和数据下标，一般用在 for 循环当中
            
            此处idx就是下标，module就是数据(Module子类的一个实例)
            见 https://www.runoob.com/python/python-func-enumerate.html
            '''
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            '''
            父类中_modules是一个     有序字典
            '''
            '''
            此处，把其存成一个    有序字典OrderedDict
            str(idx)每次迭代的变为字符串的0、1、2为key
            module为value
            '''
            self._modules[str(idx)] = module
            '''
            str函数把里面的内容转换成字符串，即作为字符串的安装迭代的0、1、2
            '''

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            '''
            self._modules.values()
            就是每层的module
            对这个做个for循环遍历，因为是OrderedDict是有顺序的
            '''
            X = block(X)
        return X

# __init__函数将每个模块逐个添加到有序字典_modules中。
# 你可能会好奇为什么每个Module都有一个_modules属性？
# 以及为什么我们使用它而不是自己定义一个Python列表？
# 简而言之，_modules的主要优点是： 在模块的参数初始化过程中， 系统知道在_modules字典中查找需要初始化参数的子块。
# 当MySequential的前向传播函数被调用时， 每个添加的块都按照它们被添加的顺序执行。
# 现在可以使用我们的MySequential类重新实现多层感知机。
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))
'''
tensor([[ 0.1097, -0.0284, -0.2538,  0.1992,  0.0850,  0.1483,  0.1369, -0.1217,
          0.1577,  0.1077],
        [ 0.0629, -0.0702, -0.2885,  0.1479,  0.1436,  0.0969,  0.1017, -0.1469,
          0.1031,  0.1670]], grad_fn=<AddmmBackward0>)
'''
# 请注意，MySequential的用法与之前为Sequential类编写的代码相同

'''
3、在前向传播函数中执行代码
'''
# Sequential类使模型构造变得简单， 允许我们组合新的架构，而不必定义自己的类。
# 然而，并不是所有的架构都是简单的顺序架构。
# 当需要更强的灵活性时，我们需要定义自己的块。
# 例如，我们可能希望在前向传播函数中执行Python的控制流。
# 此外，我们可能希望执行任意的数学运算， 而不是简单地依赖预定义的神经网络层。

# 因此我们实现了一个FixedHiddenMLP类，如下所示：
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        '''
        requires_grad=False
        即改参数不能进行梯度反传，不参与训练
        '''
        self.linear = nn.Linear(20, 20)
        '''
        定义一个层linear，这个层是一个20X20的层
        '''
    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        '''
        这里就是把X和weights一起乘，因为requires_grad=False，反传不会更新参数
        该层参数固定
        '''
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

# 在这个FixedHiddenMLP模型中，我们实现了一个隐藏层，
# 其权重（self.rand_weight）在实例化时被随机初始化，之后为常量。
# 这个权重不是一个模型参数，因此它永远不会被反向传播更新。
# 然后，神经网络将这个固定层的输出通过一个全连接层。

# 注意，在返回输出之前，模型做了一些不寻常的事情：
# 它运行了一个while循环，在L1范数大于1的条件下， 将输出向量除以2，直到它满足条件为止。
# 最后，模型返回了X中所有项的和。
# 注意，此操作可能不会常用于在任何实际任务中， 我们只是向你展示如何将任意代码集成到神经网络计算的流程中。

net = FixedHiddenMLP()
print(net(X))
'''
tensor(0.0607, grad_fn=<SumBackward0>)
'''
# 我们可以混合搭配各种组合块的方法。 在下面的例子中，我们以一些想到的方法嵌套块。
'''
即可以把nn.Sequential这种封装的神经网络嵌套到我们自定义的网络中
也可以把自定义的网络放到nn.Sequential里
即互相可以放，很灵活
'''
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))
'''
tensor(0.2321, grad_fn=<SumBackward0>)
'''
























