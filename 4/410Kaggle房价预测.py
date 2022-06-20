import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
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
print(all_features.shape)
'''
(2919, 331)
'''

# 你可以看到，此转换会将特征的总数量从79个增加到331个。
# 最后，通过values属性，我们可以 从pandas格式中提取NumPy格式，并将其转换为张量表示用于训练。
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)






