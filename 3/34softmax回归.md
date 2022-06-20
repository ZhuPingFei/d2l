![image-20220604164511616](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604164511616.png)

输出为   向量   独热编码    

![image-20220604164544883](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604164544883.png)

根据输出中最大概率的那个作为预测

![image-20220604164648726](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604164648726.png)



softmax就是用来对这个输出结果变成   概率。（和为1）

![image-20220604164818886](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604164818886.png)

做完之后，argmax返回最大值的下标，得到预测的输出结果







==？？？==

==损失函数==

使用极大似然估计对数化。即把==exp（）变成    log（）形式==



softmax对数似然

![image-20220604211113668](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604211113668.png)

小y和小x是==输出向量==和==输入向量==，Y和X是输入矩阵和输出矩阵。



直接再外面套一个  -log，==累乘也就变成累加==

则如果要最大化里面的内容就是最小化这个   -log

那么他的     损失函数







交叉熵损失函数

![image-20220604213502030](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604213502030.png)



![image-20220604213811710](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604213811710.png)

指的是其中非0即1的个数，当这个为0时，表示是无关的类别，不考虑其损失

为1则考虑其损失，此时函数为

![image-20220604213911758](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604213911758.png)

即对应的那个    真实值为1   的位置作为下标    的==损失==



-log是因为里面越大，则-log越小，以此来做优化



里面的就是

![image-20220604214116271](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604214116271.png)

那个跟   真实值中   1  对应位置   的softmax后的概率



也就是   预测结果  中对于真实预测的    概率，==这个概率越大，模型越好。==

用-log来做优化，使他最大

![image-20220604213518533](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604213518533.png)

最后这个就是把所有的损失取平均



套上exp，累加变累乘

最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。

（小标$y^{(i)}$）指的是y hat 的下标为   真实  预测的那个对应的下标，一个标量。



这个式子就是  每个样本预测真实类别的那个y hat的值的大小（概率）的累乘

即联合概率









得到了损失函数后

把  y hat  带入式子，计算

![image-20220604221027889](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604221027889.png)

因为

![image-20220604221238967](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604221238967.png)

前面是一个10组成的向量，且只有一个1

所以直接约掉



对  化简后的   损失函数  ==求梯度==

==？？？为什么是对oj求梯度而不是对w和b？？？==

![image-20220604221344344](C:\Users\ZhuPingfei\AppData\Roaming\Typora\typora-user-images\image-20220604221344344.png)

ln求导为1/x,所以分母为    log内的函数

exp求导为本身，其中只有一项为oj，其他为其他无关项相加，无导数。所以分子为exp(oj)







后半中，只有一个yj跟oj相乘，其他没有联系的常数，所以只剩下yj



==至此，求梯度推算完成==













