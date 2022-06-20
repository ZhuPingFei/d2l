import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
# 上一章中我们写了如何读取图片数据为dataloader，这里调用

if __name__=="__main__":
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


'''
初始化模型参数
'''
# 和之前线性回归的例子一样，这里的每个样本都将用固定长度的向量表示。
# 原始数据集中的每个样本都是  28×28 的图像。
# 在本节中，我们将展平每个图像，把它们看作长度为   784  的向量。

# 在后面的章节中，我们将讨论能够利用图像空间结构的特征， 但现在我们暂时只把每个像素位置看作一个特征。
#
# 回想一下，在softmax回归中，我们的输出与类别一样多。 因为我们的数据集有10个类别，所以网络输出维度为10。
# 因此，权重将构成一个 784×10  的矩阵， 偏置将构成一个 1×10  的行向量。
# 与线性回归一样，我们将使用正态分布初始化我们的权重W，偏置初始化为0。

# 因为输出的是一个向量，向量的每个值都是要靠内积算出来的，所以   每个都对应一个784长度的w向量，共10个
# 正态分布是指784内部正态分布
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

'''
定义softmax操作
'''
# 知识点回顾

#  给定一个矩阵X，我们可以对所有元素求和（默认情况下）。
#  也可以只求同一个轴上的元素，即同一列（轴0）或同一行（轴1）。
#  如果X是一个形状为(2, 3)的张量，我们对列进行求和， 则结果将是一个具有形状(3,)的向量。
#  当调用sum运算符时，我们可以指定保持在原始张量的轴数，而不折叠求和的维度。
#  这将产生一个具有形状(1, 3)的二维张量。
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True))
print( X.sum(1, keepdim=True))
'''
tensor([[5., 7., 9.]])
tensor([[ 6.],
        [15.]])
'''
# 回顾结束

# 回想一下，实现softmax由三个步骤组成：
# 1、对每个项求幂（使用exp）；
# 2、对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
# 3、将每一行除以其规范化常数，确保结果的和为1。
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
# 我们输入的是矩阵，对于矩阵，我们按照   行   做softmax
# 所以是sum维度1
# 然后用广播一除

# 对于任何随机输入，我们将每个元素变成一个非负数。
# 此外，依据概率原理，每行总和为1。

# 测试
X = torch.normal(0, 1, (2, 5))# 正态分布的25矩阵
X_prob = softmax(X)
print(X)
print(X_prob)
print(X_prob.sum(1))# 这里sum没有keepdim，所以直接少一维变成2向量
'''
tensor([[-0.9357, -1.1206, -0.4160, -1.0706,  1.7532],
        [ 0.0669, -0.8726, -1.5268, -0.1376, -0.7314]])
tensor([[0.0524, 0.0435, 0.0880, 0.0457, 0.7704],
        [0.3498, 0.1367, 0.0711, 0.2851, 0.1574]])
tensor([1., 1.])
'''

'''
定义模型
'''
# 定义softmax操作后，我们可以实现softmax回归模型。
# 下面的代码定义了输入如何通过网络映射到输出。
#
# 注:将数据传递到模型之前，我们使用reshape函数将每张原始图像展平为向量。
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
# 其中-1表示其他维度固定，这个随之改变。
# w.shape[0]为784，故Xreshape成为256(批量大小),784的矩阵
# 然后内积再加b，做softmax。最后return


'''
定义损失函数
'''
# 知识点回顾

# 接下来，我们实现 3.4节中引入的交叉熵损失函数。
# 这可能是深度学习中最常见的损失函数，因为目前分类问题的数量远远超过回归问题的数量。
#
# 回顾一下，交叉熵采用真实标签的预测概率的负对数似然。
# 这里我们不使用Python的for循环迭代预测（这往往是低效的），而是通过一个运算符选择所有元素。
# 下面，我们创建一个数据样本y_hat，其中包含2个样本在3个类别的预测概率， 以及它们对应的标签y。
# 有了y，我们知道在第一个样本中，第一类是正确的预测； 而在第二个样本中，第三类是正确的预测。
# 然后使用y作为y_hat中概率的索引， 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])
# 对于y_hat预测，第0个样本，拿出y中对应下标的元素。
# 第一个样本，拿出y中对应下标的元素
# 所以就是
# y_hat[range(len(y_hat)), y]    range是从0开始，然后给其y_hat的长度

# 注：其实这里就是一个数组取值的过程，第一维输入要的行，第二维输入要的列
'''
tensor([0.1000, 0.5000])
'''
# 回顾结束

# 现在我们只需一行代码就可以实现交叉熵损失函数。
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
# 对于取出的元素做-log
# 注：与数学表达式稍有不同，但是不影响，因为乘在外面就是用来取数字的，这里只不过把取数字在内部实现了


# 注：-log    因为我们要最大化那个对的标签的概率，也就是最大化对应位置的y_hat。
# 因为我们优化是做最小化，所以用-log来做为损失函数，最小化-log就是最大化y_hat对应位置的概率
print(cross_entropy(y_hat, y))
# 此处得到两个样本的损失
# 因为不会大于1，所以-log一定为正
'''
tensor([2.3026, 0.6931])
'''


'''
分类精度
'''
# 当预测与标签分类y一致时，即是正确的。
# 分类精度即正确预测数量与总预测数量之比。
# 虽然直接优化精度可能很困难（因为精度的计算不可导）， 但精度通常是我们最关心的性能衡量标准，我们在训练分类器时几乎总会关注它。
#
# 为了计算精度，我们执行以下操作。
# 首先，如果y_hat是矩阵，那么假定第二个维度存储每个类的预测分数。
# 我们使用argmax获得每行中最大元素的索引来获得预测类别。
# 然后我们将预测类别与真实y元素进行比较。
#
# 由于等式运算符“==”对数据类型很敏感， 因此我们将y_hat的数据类型转换为与y的数据类型一致。
# 结果是一个包含0（错）和1（对）的张量。
# 最后，我们求和会得到正确预测的数量。
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        '''
        y_hat.shape会输出一个tensor矩阵来表示其形状
        len一下则表示这个    表示形状的矩阵的大小   (这里是二维)
        即为2，表示那进来的会是一个批量的数据。（即使只有一个，也是两个维度，然后1）
        
        y_hat.shape[1]表示行有多大，即有几种预测分类，大于一才有用
        '''
        y_hat1 = y_hat.argmax(axis=1)
        # 把最大的下标存在y_hat1里，即预测的分类类别
    cmp = y_hat1.type(y.dtype) == y
    # 注：此处的y是0、2之类的格式，不是one_hot格式（[0,0,1,0]）
    # 所以用y_hat1取到下标，合为一个多个预测的向量，然后比较
    # 可能存在数据类型不同的情况，把y_hat转成y的数据类型，进行比较。传为布尔值
    return float(cmp.type(y.dtype).sum())# 把布尔值加起来，就是正确的个数。回传正确个数

# 我们将继续使用之前定义的变量y_hat和y分别作为预测的概率分布和标签。
# 可以看到，第一个样本的预测类别是2（该行的最大元素为0.6，索引为2），这与实际标签0不一致。
# 第二个样本的预测类别是2（该行的最大元素为0.5，索引为2），这与实际标签2一致。
# 因此，这两个样本的分类精度率为0.5。

# 测试
print(accuracy(y_hat, y) / len(y))
'''
0.5
'''

# 测试结束



# 同样，对于任意数据迭代器data_iter可访问的数据集， 我们可以评估在任意模型net的精度。
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式，此处不计算梯度
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:# 取输出输出
            metric.add(accuracy(net(X), y), y.numel())# 用accuracy计算正确个数，和y的numel(即预测总数)放进一个Accumulator类中
            # 这是一个累加器，每次不断的加，即变成总正确预测和总预测总数
    return metric[0] / metric[1] # 迭代结束，那么取出来累加器中数据即可

# 这里定义一个实用程序类Accumulator，用于对多个变量进行累加。
# 在上面的evaluate_accuracy函数中， 我们在Accumulator实例中创建了2个变量， 分别用于存储正确预测的数量和预测的总数量。
# 当我们遍历数据集时，两者都将随着时间的推移而累加。
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 由于我们使用随机权重初始化net模型， 因此该模型的精度应接近于随机猜测。 例如在有10个类别情况下的精度为0.1。

if __name__=="__main__": # 因为多线程程序要放在主函数中训练。写的d2l里有num_workers,是多线程
    a = evaluate_accuracy(net, test_iter)
    print(a)

'''
0.0698
'''

'''
训练
'''
# softmax回归的训练过程代码应该看起来非常眼熟。
# 在这里，我们重构训练过程的实现以使其可重复使用。
# 首先，我们定义一个函数来训练一个迭代周期。
# 请注意，updater是更新模型参数的常用函数，它接受   批量大小   作为参数。
# 它可以是d2l.sgd函数，也可以是框架的内置优化函数。
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        '''
        # 计算预测结果y_hat,是softmax之后的一个所有概率的矩阵
        '''
        l = loss(y_hat, y)
        '''
        # 计算损失，这里传入的损失是交叉熵损失，得到样本的损失给l
        '''

        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            '''
            # 这里的updater是SDG算法，即    小批量随机梯度下降
            '''

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        '''
        这里保存了  损失总和  和    准确预测总和      和       样本总和
        '''

    # 返回训练损失和训练精度
    '''
    用损失总和除以样本个数，就是平均损失。(此处的返回跟SGD梯度下降处无关，那里已经用sum来计算梯度了)
    '''
    return metric[0] / metric[2], metric[1] / metric[2]



# 在展示训练函数的实现之前，我们定义一个在动画中绘制数据的实用程序类Animator， 它能够简化本书其余部分的代码
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw();
        plt.pause(0.001)
        display.display(self.fig)
        display.clear_output(wait=True)

# 接下来我们实现一个训练函数， 它会在train_iter访问到的训练数据集上训练一个模型net。
# 该训练函数将会运行多个迭代周期（由num_epochs指定）。
# 在每个迭代周期结束时，利用test_iter访问到的测试数据集对模型进行评估。
# 我们将利用Animator类来可视化训练进度。
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    '''定义可视化'''
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    '''
    进行多个epoch
    '''
    for epoch in range(num_epochs):
        '''训练一次，并把返回的数据放到train_metrics里，训练损失、训练精度'''
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        '''在测试数据集上评估精度'''
        test_acc = evaluate_accuracy(net, test_iter)
        '''加训练的loss、精度和测试的精度进去'''
        animator.add(epoch + 1, train_metrics + (test_acc,))
    '''把训练的loss和精度给分别两个值'''
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
# assert 使用assert可以在出现有异常的代码处直接终止运行。 而不用等到程序执行完毕之后抛出异常。
# 表达式=false 时，则执行其后面的异常
# 见 https://www.php.cn/python-tutorials-416728.html
# 和 https://blog.csdn.net/qq_37369201/article/details/109195257



# 作为一个从零开始的实现，我们使用 3.2节中定义的 小批量随机梯度下降来优化模型的损失函数，设置学习率为0.1。
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


# 现在，我们训练模型10个迭代周期。
# 请注意，迭代周期（num_epochs）和学习率（lr）都是可调节的超参数。
# 通过更改它们的值，我们可以提高模型的分类精度。
if __name__=="__main__":
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)



'''
预测
'''
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
        # 这里的break指我们只拿出来一个样本
    # 这里指拿出         正确的标签名        和        预测的标签名
    # 这里调用的是35写的，用   预测数字（拿去当下标）  获得标签名的返回
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))

    # 制作title然后plt里展示
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()

if __name__=="__main__":
    predict_ch3(net, test_iter)




