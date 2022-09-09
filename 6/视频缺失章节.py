'''
视频缺失章节
'''

'''
池化层的输出通道数和输入通道数相同
'''


'''
# 卷积神经网络中卷积核的高度和宽度通常为奇数，例如1、3、5或7。 
# 选择奇数的好处是，保持空间维度的同时，我们可以在顶部和底部填充相同数量的行，在左侧和右侧填充相同数量的列。
'''

'''
再选择填入的kernel_size和padding时候，如果是int那就是正方形的核以及宽高填入相同的行列(上下左右各填充那么多)
如果是元组，则    先高后宽，上下各几行左右各几列

stride 步幅同理，int就是以 这个数 在宽高上作为步幅
元组就是  先高后宽
'''

'''
zip会把可迭代对象作为参数，将对象中对应元素打包成一个个元组，然后返回由这些元素组成的列表
即把最外层[]拆了，然后把对应的x和k打包成元组，好几个元组组成列表。for列表取元组，然后分别给到这轮的x，k
见https://mofanpy.com/tutorials/python-basic/interactive-python/lazy-usage/#Zip%E8%AE%A9%E4%BD%A0%E5%90%8C%E6%97%B6%E8%BF%AD%E4%BB%A3
'''

'''
torch.stack：沿着一个新维度(此处为0维开始连接，即直接前后concat)，对张量序列(接收一个张量列表作为输入)进行连接，序列中所有张量应为相同维度

所以concat的操作为
torch.stack(张量列表,0)

'''

'''
用pytorch的封好的写法，就是nn.Conv2d中前两个维度指定输出输入,
不过封装好的函数中是先输入后输出
'''


'''
默认情况下，深度学习框架中的步幅与池化窗口的大小相同。
即如果2X2的池化窗口，步幅为2。窗口移动没有重叠
填充和步幅可以手动设定。
'''


'''
矩阵合并可以用torch.cat
上面那个是因为生成的结果存在列表里面，所以用上面的stack方法
'''


'''
train 和  eval 在前向中有不同
model.train()的作用是启用 Batch Normalization 和 Dropout。
model.eval()的作用是不启用 Batch Normalization 和 Dropout。
model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

BN层就是压缩数据的区间
'''


'''
    with torch.no_grad():
        
        见：https://blog.csdn.net/sazass/article/details/116668755
        
'''