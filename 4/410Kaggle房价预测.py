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


















