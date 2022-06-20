import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
# 用SVG显示图片，清晰度高一点

'''
读取数据集
'''
# 通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中。
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()# 把图片转换成torch.tensor格式，把这个函数给到trans，之后可以用trans来做这个函数

# transform=trans  表示对图片做这个操作，拿到以后的图片是tensor格式
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)# 下载训练集（train = true）
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)# 下载测试集（train = true）

# Fashion-MNIST由10个类别的图像组成， 每个类别由训练数据集（train dataset）中的6000张图像 和测试数据集（test dataset）中的1000张图像组成。
# 因此，训练集和测试集分别包含60000和10000张图像。
# 测试数据集不会用于训练，只用于评估模型性能。
print(len(mnist_train), len(mnist_test))
'''
60000 10000
'''
print(mnist_train[0][0].shape)
# 第一个0是第0个example，第二个0是指第0张图片
# shape输出其形状
'''
torch.Size([1, 28, 28])
'''


# Fashion-MNIST中包含的10个类别
# 分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。

# 以下函数用于在    数字标签索引   及    其文本名称    之间进行转换。
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 我们现在可以创建一个函数来可视化这些样本。
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())

        else:
            # PIL图片
            ax.imshow(img)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# 以下是训练数据集中前几个样本的图像及其相应的标签。
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
# 把图片本身的channel数不要了，换成批量数18。即本来X是一个(18,1,28,28)变成(18,28,28)。（本来那里也只有一层，直接不要了）
# 把其reshape成一个numble of examples
# 画两行，每行9张图片，传入title
# 注：这个不是matplotlib的内置函数，这是我们构造的，matplotlib具体的在上面函数中
d2l.plt.show()

'''
读取小批量
'''
# 为了使我们在读取训练集和测试集时更容易，我们使用内置的数据迭代器，而不是从零开始创建。
# 回顾一下，在每次迭代中，数据加载器每次都会读取一小批量数据，大小为batch_size。
# 通过内置数据迭代器，我们可以随机打乱了所有样本，从而无偏见地读取小批量。
batch_size = 256

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
# num_workers是用多少个进程

# 我们看一下读取训练数据所需的时间。
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
# 字符串前加 f 的含义：格式化 {} 内容，不在 {} 内的照常展示输出，如果你想输出 {}，那就用双层 {{}} 将想输出的内容包起来。
# 也就是之后字符串中{}内的内容，输出为其值
# 见https://blog.csdn.net/qq_43463045/article/details/93890436
'''
2.60 sec
'''


'''
整合所有组件
'''
# 现在我们定义load_data_fashion_mnist函数，用于获取和读取Fashion-MNIST数据集。
# 这个函数返回训练集和验证集的        数据迭代器。
# 此外，这个函数还接受一个可选参数resize，用来将图像大小调整为另一种形状。
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
        # python insert()函数用于将指定对象插入列表的指定位置
        # 第一个参数为索引位置，第二个参数为对象
        # 插入后，后面的往后移一位
        # 见 https://blog.csdn.net/lizhaoyi123/article/details/102690504
    trans = transforms.Compose(trans)
    # transforms.Compose(trans)串联多个图片变换的操作。
    # Compose()类会将trans列表里面的transform操作进行遍历
    # 见https://blog.csdn.net/b_dxac/article/details/115611780
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

# 下面，我们通过指定resize参数来测试load_data_fashion_mnist函数的图像大小调整功能。
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

# 注：这里for一下就break，表示我们只取一个然后输出就结束循环
'''
torch.Size([32, 1, 64, 64]) torch.float32 torch.Size([32]) torch.int64
'''



























