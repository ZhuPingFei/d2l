import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
生成数据集
'''
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# 这个d2l的相当于我们3.2写的那个synthetic_data函数
# 这里就直接用了不重写，自己做还是要重写

'''
读取数据集
'''
# 我们可以调用框架中现有的API来读取数据。
# 我们将features和labels作为API的参数传递，并通过数据迭代器指定batch_size。
# 此外，布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据。
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    # 把features和labels组合成一个list传到TensorDataset中
    # 得到pytorch的一个dataset
    dataset = data.TensorDataset(*data_arrays)
    # 功能为：把tensor打包，一一对应，其第一维个数一致(如此处（1000，2）和（1000,1）)
    # 见https://blog.csdn.net/anshiquanshu/article/details/109398797
    # 和https://zhuanlan.zhihu.com/p/349083821
    # 星号，作用是将调用时提供的所有值，放在一个元组里。
    # 见https://blog.csdn.net/zkk9527/article/details/88675129
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    # 得到torch的dataset后，可以把dataset和batchsize和是否打乱  传进   dataloader中
    # batchsize每次调用几个，shuffle为是否打乱(如果是训练，则需要打乱)
# 返回的是一个iterator数据迭代器

batch_size = 10
data_iter = load_array((features, labels), batch_size)# 把数据迭代器赋给data_iter

# 使用data_iter的方式与我们在 3.2节中使用data_iter函数的方式相同。
# 为了验证是否正常工作，让我们读取并打印第一个小批量样本。


# 与 3.2节不同，这里我们使用iter构造Python迭代器
# 并使用next从迭代器中获取第一项。
print(next(iter(data_iter)))
'''
[tensor([[ 0.7146, -1.0284],
        [ 0.6692,  0.6394],
        [ 0.9853,  0.9755],
        [-0.1454,  1.4553],
        [ 0.2490, -0.0134],
        [ 0.8892,  0.4315],
        [-0.3187, -0.2522],
        [ 2.5774, -1.2065],
        [-0.0783,  0.0147],
        [-1.0910,  0.4044]]), tensor([[ 9.1257],
        [ 3.3670],
        [ 2.8481],
        [-1.0384],
        [ 4.7476],
        [ 4.4950],
        [ 4.4207],
        [13.4565],
        [ 3.9915],
        [ 0.6342]])]
'''

'''
定义模型
'''
# nn是神经网络的缩写

# 对于标准深度学习模型，我们可以使用框架的预定义好的层。
# 这使我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。

# 我们首先定义一个   模型变量   net，它是一个Sequential类的实例。
# Sequential类将多个层串联在一起。
# 当给定输入数据时，Sequential实例  将数据  传入到第一层，然后将第一层的输出作为第二层的输入，以此类推。
# 在下面的例子中，我们的模型只包含一个层，因此实际上不需要Sequential。
# 但是由于以后几乎所有的模型都是多层的，在这里使用Sequential会让你熟悉“标准的流水线”。

# 在PyTorch中，全连接层在Linear类中定义。
# 值得注意的是，我们将两个参数传递到nn.Linear中。

# 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))# 使用全连接层，使用一层矩阵向量乘法，指定输入为2维输出为1维。
# 放到一个sequential的容器里面，可以理解为list of layers，把好几个层按顺序放在一起

'''
初始化模型参数
'''
# 在使用net之前，我们需要初始化模型参数。
# 如在线性回归模型中的权重和偏置。
# 深度学习框架通常有预定义的方法来初始化参数。
# 在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 偏置参数将初始化为零。

# 正如我们在构造nn.Linear时指定输入和输出尺寸一样， 现在我们能直接访问参数以设定它们的初始值。
# 我们通过net[0]选择网络中的第一个图层，
# 然后使用weight.data和bias.data方法访问参数。
# 我们还可以使用替换方法normal_和fill_来重写参数值。

net[0].weight.data.normal_(0, 0.01) # 给weight写一个正态分布
net[0].bias.data.fill_(0) # 用fill来补充0

# 应该是线性全连接层本身  对于x输入有对应个数的weight（相当于w）和bias（相当于b），这里就是给其初始化了，不用像之前自己从零做一个初值w和b
# 网络本身有初始值，这里只是为了跟前一节对应。可以不另设初始值。
'''
定义损失函数
'''
# 计算均方误差使用的是MSELoss类，也称为平方 L2 范数。
# 默认情况下，它返回  所有样本损失的平均值。
loss = nn.MSELoss()
'''
注：返回的是差的平方，即(x-y)的平方
然后取平均
'''

'''
定义优化算法
'''
# 小批量随机梯度下降算法是一种优化神经网络的标准工具， PyTorch在optim模块中实现了该算法的许多变种。
# 当我们实例化一个SGD实例时，我们要指定
#
# 1、优化的参数 （可通过net.parameters()从我们的模型中获得）
# 以及
# 2、优化算法所需的超参数字典。

# 小批量随机梯度下降的   超参数  只需要设置lr值（学习率），这里设置为0.03。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# SGD就是   小批量随机梯度下降  ，逻辑见3.2。即把小批量的梯度作为全局的梯度来更新参数。


'''
训练
'''
# 通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码。
# 我们不必单独分配参数、不必定义我们的损失函数，也不必手动实现小批量随机梯度下降。
# 当我们需要更复杂的模型时，高级API的优势将大大增加。 当我们有了所有的基本组件，训练过程代码与我们从零开始实现时所做的非常相似。
#
# 回顾一下：在每个迭代周期里，我们将完整遍历一次数据集（train_data）， 不停地从中获取一个小批量的输入和相应的标签。 对于每一个小批量，我们会进行以下步骤:
# 通过调用net(X)生成预测并计算损失l（前向传播）。
# 通过进行反向传播来计算梯度。
# 通过调用优化器来更新模型参数。
# 为了更好的衡量训练效果，我们计算每个迭代周期后的损失，并打印它来监控训练过程。
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y) # 把X放到net中，此时的net已经在前面定义了模型参数，所以不用放w和b
        # 此时的loss也是会自己返回所有样本损失的平均值
        trainer.zero_grad()# 优化器先把梯度清零
        l.backward()# 计算梯度，梯度保存在模型的weight.grad和bias.grad中
        trainer.step()# 进行模型参数的更新，就是更新net中的weight和bias.(调用他们自己的grad)
    # 每一个epoch，用loss显示一下效果
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
'''
epoch 1, loss 0.000254
epoch 2, loss 0.000101
epoch 3, loss 0.000101
'''

# 下面我们比较生成数据集的真实参数和通过有限数据训练获得的模型参数。
# 要访问参数，我们首先从net访问所需的层，然后读取该层的权重和偏置。
# 正如在从零开始实现中一样，我们估计得到的参数与生成数据的真实参数非常接近。
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
'''
w的估计误差： tensor([-0.0003, -0.0008])
b的估计误差： tensor([0.0006])
'''













