import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from IPython import display
from matplotlib import pyplot as plt
'''
一、下载和缓存数据集
这个不用太明白，跟所学无关，为了下载数据提供了个脚本罢了
'''
# 在整本书中，我们将下载不同的数据集，并训练和测试模型。
# 这里我们实现几个函数来方便下载数据。
# 首先，我们建立字典DATA_HUB， 它可以将数据集名称的字符串映射到数据集相关的二元组上， 这个二元组包含数据集的url和验证文件完整性的sha-1密钥。
# 所有类似的数据集都托管在地址为DATA_URL的站点上。
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import hashlib
import os
import tarfile
import zipfile
import requests

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 下面的download函数用来下载数据集， 将数据集缓存在本地目录（默认情况下为../data）中， 并返回下载文件的名称。
# 如果缓存目录中已经存在此数据集文件，并且其sha-1与存储在DATA_HUB中的相匹配， 我们将使用缓存的文件，以避免重复的下载。

def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


# 我们还需实现两个实用函数：
# 一个将下载并解压缩一个zip或tar文件，
# 另一个是将本书中使用的所有数据集从DATA_HUB下载到缓存目录中。
def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)




'''
二、访问和读取数据集
知道调用了函数下载或加载了数据集到train_data 和 test_data 就行
'''
# 注意，竞赛数据分为训练集和测试集。
# 每条记录都包括房屋的属性值和属性，如街道类型、施工年份、屋顶类型、地下室状况等。
# 这些特征由各种数据类型组成。
# 例如，建筑年份由整数表示，屋顶类型由离散类别表示，其他特征由浮点数表示。
# 这就是现实让事情变得复杂的地方：例如，一些数据完全丢失了，缺失值被简单地标记为“NA”。
# 每套房子的价格只出现在训练集中（毕竟这是一场比赛）。
# 我们将希望划分训练集以创建验证集，但是在将预测结果上传到Kaggle之后， 我们只能在官方测试集中评估我们的模型。

# 开始之前，我们将使用pandas读入并处理数据， 这是我们在 2.2节中引入的。
'''
为方便起见，我们可以使用上面定义的脚本下载并缓存Kaggle房屋数据集。
'''

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# 我们使用pandas分别加载包含训练数据和测试数据的两个CSV文件。
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# 训练数据集包括1460个样本，每个样本80个特征和1个标签， 而测试数据集包含1459个样本，每个样本80个特征。
print(train_data.shape)
print(test_data.shape)
'''
(1460, 81)
(1459, 80)
'''

# 让我们看看前四个和最后两个特征，以及相应标签（房价）。
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
'''
   Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice
0   1          60       RL         65.0       WD        Normal     208500
1   2          20       RL         80.0       WD        Normal     181500
2   3          60       RL         68.0       WD        Normal     223500
3   4          70       RL         60.0       WD       Abnorml     140000
'''
# 我们可以看到，在每个样本中，第一个特征是ID， 这有助于模型识别每个训练样本。
# 虽然这很方便，但它不携带任何用于预测的信息。
# 因此，在将数据提供给模型之前，我们将其从数据集中删除。

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

'''
此处的all_features,指的是所有的参数。
所以训练集排除第一列id和最后一列正确预测房价标签
测试集排除第一列id

pd.concat数据拼接

见https://zhuanlan.zhihu.com/p/132593960
'''


'''
三、数据预处理
'''

# 在开始建模之前，我们需要对数据进行预处理。
# 首先，我们将所有缺失的值替换为相应特征的平均值。
# 然后，为了将所有特征放在一个共同的尺度上，
# 我们通过将特征重新缩放到  零均值   和   单位方差   来标准化数据：
# 见md 解释数学概念
# 直观地说，我们标准化数据有两个原因：
# 首先，它方便优化。
# 其次，因为我们不知道哪些特征是相关的， 所以我们不想让惩罚分配给一个特征的系数比分配给其他任何特征的系数更大。


# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
print(all_features.dtypes)
'''
MSSubClass         int64
MSZoning          object
LotFrontage      float64
LotArea            int64
Street            object
                  ...   
MiscVal            int64
MoSold             int64
YrSold             int64
SaleType          object
SaleCondition     object
Length: 79, dtype: object
'''
print(numeric_features)
'''
Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold'],
      dtype='object')
'''
'''
从all_features.dtypes中拿出dtype不是object的列的index(标题)
赋值给numeric_features

实现逻辑：all_features.dtypes会返回一个index标题和其数据类型的二维数组
all_features.dtypes != 'object' 会返回一个位置一一对应的true false
这样会挑选出一个新的类似all_features.dtypes但是没有object的二维数组
然后调用.index方法取出index

dtypes见https://zhuanlan.zhihu.com/p/350568058
index见https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.html
'''
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
'''
对于所有的数值，我们用所拿到的是数值的标题来取这几列，
应用一个函数，这个函数是见md解释里面的意思
使其所有数据变成均值为0方差为1。

此处是把训练集和测试集放在一起做均值和方差
'''
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
'''
fillna(0)在实际数据中，对应的是数值的那几列，把Nan改成0
(因为经过我们前面的归一化，平均值已经变成0了)
'''

# 接下来，我们处理离散值。
'''
即对字符串编码
如果RL表示一类、RM表示一类
用onehot为其编码
'''
# 这包括诸如“MSZoning”之类的特征。
# 我们用独热编码替换它们， 方法与前面将多类别标签转换为向量的方式相同 （请参见 3.4.1节）。
'''
如y1=(0,0,1)y=(0,1,0)y=(1,0,0)
'''
# 例如，“MSZoning”包含值“RL”和“Rm”。 我们将创建两个新的指示器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1。
# 根据独热编码，如果“MSZoning”的原始值为“RL”， 则：“MSZoning_RL”为1，“MSZoning_RM”为0。
# pandas软件包会自动为我们实现这一点。

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
'''
可以见22数据预处理40行开始，这里把缺省数据列入考量
'''
print(all_features.shape)
'''
(2919, 331)
'''
# 你可以看到，此转换会将特征的总数量从79个增加到331个。
'''
onehot后，好多object字段的特征一分为多，特征总数从79变成了331个
'''


# 最后，通过values属性，我们可以 从pandas格式中提取NumPy格式，并将其转换为张量表示用于训练。
n_train = train_data.shape[0]
print(n_train)
'''
1460
'''

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

'''
默认float64，这里指定为32

train_data.shape是一个表示形状的张量，这里是二维张量
train_data.shape[0]就是其0维，也就是1460,81中的1460

torch.tensor(all_features[:n_train].values

就是取前1460行作为训练集的values，合为一个二维列表(通过torch.tensor变成tensor格式)
(与之相反的是index标题)

所以测试集就是all_features[n_train:]
'''

'''
trainlabels,这个就是房价金额，这是标签(输出值)
去traindata的SalePrice这个index的values，并且reshape
使得数据每行一个，自适应行数

可以与前面的参数一一对照
'''

'''
四、训练
'''
# 首先，我们训练一个带有损失平方的线性模型。
# 显然线性模型很难让我们在竞赛中获胜，但线性模型提供了一种健全性检查， 以查看数据中是否存在有意义的信息。
# 如果我们在这里不能做得比随机猜测更好，那么我们很可能存在数据处理错误。
# 如果一切顺利，线性模型将作为基线（baseline）模型， 让我们直观地知道最好的模型有超出简单的模型多少。
loss = nn.MSELoss()
'''
差的平方
取平均值
'''
in_features = train_features.shape[1]
'''
in_features,输入的维度，331
(即train_feature的shape的第二位，第一位是数据个数)
'''

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    '''
    一个单层线性回归，输入维度331，输出维度1
    '''
    return net

# 房价就像股票价格一样，我们关心的是相对数量，而不是绝对数量。
# 因此，我们更关心相对误差 而不是  绝对误差
# 见md

# 解决这个问题的一种方法是用价格预测的对数来衡量差异。
# 使用log的均方根误差
# 见md
'''
对于比较大的值正的值做相对误差的回归时候，
'''
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    '''
    clamp
    对于net(features)，即网络的输出。
    最小值为1，小于1设置为1.
    因为小于1取log就是负数了，我们要都是正数
    不设上限，上限为float('inf')
    其值给到clipped_preds
    见 https://blog.csdn.net/hb_learing/article/details/115696206
    '''
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    '''
    这就是转变后的输出与label求均方误差loss。
    即求出 差，然后平方，然后平均
    是一个单元素tensor
    所以return时候要sqrt开根号
    概念见md
    '''
    '''
    rmse是一个单元素的tensor张量
    .item是用于把单元素张量变成数字的，保留原类型
    见 https://www.jianshu.com/p/79da0eac5f01
    https://blog.csdn.net/weixin_44739213/article/details/108659763
    使得返回的loss是一个数字
    '''
    return rmse.item()

# 与前面的部分不同，我们的训练函数将借助Adam优化器 （我们将在后面章节更详细地描述它）。
# Adam优化器的主要吸引力在于它对初始学习率不那么敏感。
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    '''
    改用adam优化算法，可以认为是一个比较平滑的SGD
    对学习率不敏感
    '''
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    '''
    weight_decay是权重衰退，输入lambda来增加惩罚项
    '''
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        '''
        下面这些是用来画图的，存在数组里
        '''
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


'''
五、K折交叉验证
'''
'''
数据集不够时，采用K折交叉验证。
这里，原始训练数据被分成k个不重叠的子集。
然后执行k次模型训练和验证，每次在k-1个子集上进行训练，并在剩余的一个子集（在该轮中没有用于训练的子集）上进行验证。
最后，通过对k次实验的结果取平均来估计训练和验证误差。
'''
# 其有助于模型选择和超参数调整。
# 我们首先需要定义一个函数，在k折交叉验证过程中返回第i折的数据。
# 具体地说，它选择第i个切片作为验证数据，其余部分作为训练数据。
# 注意，这并不是处理数据的最有效方法，如果我们的数据集大得多，会有其他解决办法。
'''
用来被调用来建立k折交叉验证的数据集
'''
def get_k_fold_data(k, i, X, y):
    '''
    给定数据集X，y。和共k折和当前折。
    返回当前的训练集和交叉验证集
    k: 分成k折
    i: 此时是第几折
    Xy:训练数据
    '''
    assert k > 1
    '''
    k不大于1就报错
    '''
    fold_size = X.shape[0] // k
    '''
    样本数除以k获得折的大小，这里是整数除法，除出来无小数
    '''
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        '''
        j * fold_size的到当前的位置
        并用(j * fold_size, (j + 1) * fold_size)取一个fold_size大小
        
        slice函数接收参数：起始点和结束点和步长
        用 tensor[slice]来取元素
        见 https://www.w3school.com.cn/python/ref_func_slice.asp
        '''
        X_part, y_part = X[idx, :], y[idx]
        '''
        tensor中第一层是取对应位置切片(对应行，即对应的数据集的部分)
        第二层取全部，指一整行
        '''
        if j == i:
            X_valid, y_valid = X_part, y_part
            '''
            j从0到k，j==i
            表示到了当前这一折，这折的数据为交叉验证集，赋值
            '''
        elif X_train is None:
            X_train, y_train = X_part, y_part
            '''
            跟i不一致且train还是空时，赋值
            '''
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
            '''
            不为空且也不是i时，粘贴上去
            即两个数据集第0维相连，就是前后相连。直接增加行数变成一个
            见 https://blog.csdn.net/xinjieyuan/article/details/105208352
            '''
    return X_train, y_train, X_valid, y_valid

# 当我们在k折交叉验证中训练k次后，返回训练和验证误差的平均值。
'''
开始做k折交叉验证了
输入训练集等等参数
返回训练和验证误差的平均值
'''
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    '''
    训练集和测试集的损失的和
    '''
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        '''
        调用上面函数
        来生成k折一次计算，即i从0到k，每一轮取一个
        产生一次数据集，包起来给data
        用data点  来访问
        '''
        net = get_net()
        '''
        net
        用一层线性
        '''
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        '''
        *data是解码，变成前面四个拿到的数据
        X_train y_train
        X_valid y_valid
        刚好和train函数要求的一致
        即  训练集参数  训练集结果
        测试集参数(验证集) 测试集结果(验证集)
        '''
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        '''
        把每一折上面的  最终的(训练到最后的)  loss求和，在最后除以k做平均作为返回值
        因为训练函数中会把每个epoch的loss保存在这个数组中(用来画图)
        所以取的最好的loss在最后一个，用-1
        '''
        '''
        画图用(其实只花了第一折的时候的图随epoch)
        '''
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')

        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


'''
六、模型选择
'''
# 在本例中，我们选择了一组未调优的超参数，并将其留给读者来改进模型。
# 找到一组调优的超参数可能需要时间，这取决于一个人优化了多少变量。
# 有了足够大的数据集和合理设置的超参数，k折交叉验证往往对多次测试具有相当的稳定性。
# 然而，如果我们尝试了不合理的超参数，我们可能会发现验证效果不再代表真正的误差。
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
d2l.plt.show()
'''
折1，训练log rmse0.169830, 验证log rmse0.156834
折2，训练log rmse0.162307, 验证log rmse0.190018
折3，训练log rmse0.164326, 验证log rmse0.168819
折4，训练log rmse0.167602, 验证log rmse0.154597
折5，训练log rmse0.162626, 验证log rmse0.182676
5-折验证: 平均训练log rmse: 0.165338, 平均验证log rmse: 0.170589
'''
# 请注意，有时一组超参数的训练误差可能非常低，但k折交叉验证的误差要高得多，这表明模型过拟合了。
# 在整个训练过程中，你将希望监控训练误差和验证误差这两个数字。
# 较少的过拟合可能表明现有数据可以支撑一个更强大的模型，较大的过拟合可能意味着我们可以通过正则化技术来获益。


'''
七、提交你的Kaggle预测
'''
# 既然我们知道应该选择什么样的超参数， 我们不妨使用所有数据对其进行训练
# （而不是仅使用交叉验证中使用的1-1/K的数据）。
# 然后，我们通过这种方式获得的模型可以应用于测试集。
# 将预测保存在CSV文件中可以简化将结果上传到Kaggle的过程。
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    d2l.plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()

    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    '''
    
    preds是一个每行一个数据，共好多行的数据集
    先reshape成为   1行自适应，这样就是单独数据不分层了，但是有两层[]因为，1也是一个维度
    所以之后[0]取出第一行所有数据
    这样就是一个队列了
    重新创建一个pandas
    即给其序列放上index(最前面有01234这种行的编码)
    见https://blog.csdn.net/TeFuirnever/article/details/94331545
    
    test_data是前面最开始readcsv最原始的pandas数据，是有标题有前面01234的数据
    我们当时训练和交叉验证的时候，是做了处理然后.values
    所以没有前面01234的行的编码
    
    现在要把数据加进去，所以要转变成有行的编码的形式，即pandas数据的形式
    '''
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    '''
    axis=1各行合并
    '''
    submission.to_csv('submission.csv', index=False)
    '''
    pandas变成csv文件
    '''


# 如果测试集上的预测与K倍交叉验证过程中的预测相似,那就是时候把它们上传到Kaggle了。
# 下面的代码将生成一个名为submission.csv的文件。
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
'''
训练log rmse：0.162347
'''




