---
title: PyTorch基础
tags:
  - 机器学习
  - Inversion
  - Deep Learning
  - PyTorch
categories:
  - 学习笔记
abbrlink: fee729e7
date: 2021-12-31 10:43:02
mathjax:
copyright:
password:
---

没想到我也有开始炼丹的一天

<!-- more -->

主要的参考资料是[这个](https://github.com/zergtant/pytorch-handbook)

# PyTorch基础

## 张量

numpy中的基础单位是ndarray，PyTorch里面的基础单位是tensor，tensor可以在gpu里进行计算

基本操作都一样，比如ones_like,zeros,size，索引方式也一样

基本的数据类型有五种

- 32位浮点型：torch.FloatTensor。 (默认)
- 64位整型：torch.LongTensor。
- 32位整型：torch.IntTensor。
- 16位整型：torch.ShortTensor。
- 64位浮点型：torch.DoubleTensor。

除以上数字类型外，还有 byte和chart型

### 张量的一些特殊操作

加法1:

```python
y = torch.rand(5, 3)
print(x + y)
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

加法2

```python
print(torch.add(x, y))
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

提供输出tensor作为参数

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

替换

```python
# adds x to y
y.add_(x)
print(y)
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

#### Note

任何 以``_`` 结尾的操作都会用结果替换原变量. 例如: ``x.copy_(y)``, ``x.t_()``, 都会改变 ``x``.

改变维度

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  #  size -1 从其他维度推断
print(x.size(), y.size(), z.size())
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

如果你有只有一个元素的张量，使用`.item()`来得到Python数据类型的数值

```python
x = torch.randn(1)
print(x)
print(x.item())
tensor([-0.2368])
-0.2368014901876 b4496
```

### 与numpy的转换

Torch Tensor与NumPy数组共享底层内存地址，修改一个会导致另一个的变化。

将一个Torch Tensor转换为NumPy数组

```python
a = torch.ones(5)
print(a)
tensor([1., 1., 1., 1., 1.])
```

```python
b = a.numpy()
print(b)
[1. 1. 1. 1. 1.]
```

观察numpy数组的值是如何改变的。

```python
a.add_(1)
print(a)
print(b)
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

NumPy Array 转化成 Torch Tensor

使用from_numpy自动转化

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

所有的 Tensor 类型默认都是基于CPU， CharTensor 类型不支持到 NumPy 的转换.

使用`.to` 方法 可以将Tensor移动到任何设备中

```python
# is_available 函数判断是否有cuda可以使用
# ``torch.device``将张量移动到指定的设备中
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
    x = x.to(device)                       # 或者直接使用``.to("cuda")``将张量移动到cuda中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 也会对变量的类型做更改
tensor([0.7632], device='cuda:0')
tensor([0.7632], dtype=torch.float64)
```

## 自动求导

没太看懂这里是用来干嘛的

## 神经网络包&优化器

torch.nn是专门为神经网络设计的模块化接口。nn构建于 Autograd之上，可用来定义和运行神经网络。 这里我们主要介绍几个一些常用的类

**约定：torch.nn 我们为了方便使用，会为他设置别名为nn，本章除nn以外还有其他的命名约定**

```python
# 首先要引入相关的包
import torch
# 引入torch.nn并指定别名
import torch.nn as nn
#打印一下版本
torch.__version__
```

```bash
'1.0.0'
```

除了nn别名以外，我们还引用了nn.functional，这个包中包含了神经网络中使用的一些常用函数，这些函数的特点是，不具有可学习的参数(如ReLU，pool，DropOut等)，这些函数可以放在构造函数中，也可以不放，但是这里建议不放。

一般情况下我们会**将nn.functional 设置为大写的F**，这样缩写方便调用

```python
import torch.nn.functional as F
```

### 定义一个网络

PyTorch中已经为我们准备好了现成的网络模型，只要继承nn.Module，并实现它的forward方法，PyTorch会根据autograd，自动实现backward函数，在forward函数中可使用任何tensor支持的函数，还可以使用if、for循环、print、log等Python语法，写法和标准的Python写法一致。

In [3]:

```python
class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        
        # 卷积层 '1'表示输入图片为单通道， '6'表示输出通道数，'3'表示卷积核为3*3
        self.conv1 = nn.Conv2d(1, 6, 3) 
        #线性层，输入1350个特征，输出10个特征
        self.fc1   = nn.Linear(1350, 10)  #这里的1350是如何计算的呢？这就要看后面的forward函数
    #正向传播 
    def forward(self, x): 
        print(x.size()) # 结果：[1, 1, 32, 32]
        # 卷积 -> 激活 -> 池化 
        x = self.conv1(x) #根据卷积的尺寸计算公式，计算结果是30，具体计算公式后面第二章第四节 卷积神经网络 有详细介绍。
        x = F.relu(x)
        print(x.size()) # 结果：[1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2)) #我们使用池化层，计算结果是15
        x = F.relu(x)
        print(x.size()) # 结果：[1, 6, 15, 15]
        # reshape，‘-1’表示自适应
        #这里做的就是压扁的操作 就是把后面的[1, 6, 15, 15]压扁，变为 [1, 1350]
        x = x.view(x.size()[0], -1) 
        print(x.size()) # 这里就是fc1层的的输入1350 
        x = self.fc1(x)        
        return x

net = Net()
print(net)
```

```bash
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=1350, out_features=10, bias=True)
)
```

网络的可学习参数通过net.parameters()返回

```python
for parameters in net.parameters():
    print(parameters)
```

```bash
Parameter containing:
tensor([[[[ 0.2745,  0.2594,  0.0171],
          [ 0.0429,  0.3013, -0.0208],
          [ 0.1459, -0.3223,  0.1797]]],


        [[[ 0.1847,  0.0227, -0.1919],
          [-0.0210, -0.1336, -0.2176],
          [-0.2164, -0.1244, -0.2428]]],


        [[[ 0.1042, -0.0055, -0.2171],
          [ 0.3306, -0.2808,  0.2058],
          [ 0.2492,  0.2971,  0.2277]]],


        [[[ 0.2134, -0.0644, -0.3044],
          [ 0.0040,  0.0828, -0.2093],
          [ 0.0204,  0.1065,  0.1168]]],


        [[[ 0.1651, -0.2244,  0.3072],
          [-0.2301,  0.2443, -0.2340],
          [ 0.0685,  0.1026,  0.1754]]],


        [[[ 0.1691, -0.0790,  0.2617],
          [ 0.1956,  0.1477,  0.0877],
          [ 0.0538, -0.3091,  0.2030]]]], requires_grad=True)
Parameter containing:
tensor([ 0.2355,  0.2949, -0.1283, -0.0848,  0.2027, -0.3331],
       requires_grad=True)
Parameter containing:
tensor([[ 2.0555e-02, -2.1445e-02, -1.7981e-02,  ..., -2.3864e-02,
          8.5149e-03, -6.2071e-04],
        [-1.1755e-02,  1.0010e-02,  2.1978e-02,  ...,  1.8433e-02,
          7.1362e-03, -4.0951e-03],
        [ 1.6187e-02,  2.1623e-02,  1.1840e-02,  ...,  5.7059e-03,
         -2.7165e-02,  1.3463e-03],
        ...,
        [-3.2552e-03,  1.7277e-02, -1.4907e-02,  ...,  7.4232e-03,
         -2.7188e-02, -4.6431e-03],
        [-1.9786e-02, -3.7382e-03,  1.2259e-02,  ...,  3.2471e-03,
         -1.2375e-02, -1.6372e-02],
        [-8.2350e-03,  4.1301e-03, -1.9192e-03,  ..., -2.3119e-05,
          2.0167e-03,  1.9528e-02]], requires_grad=True)
Parameter containing:
tensor([ 0.0162, -0.0146, -0.0218,  0.0212, -0.0119, -0.0142, -0.0079,  0.0171,
         0.0205,  0.0164], requires_grad=True)
```

net.named_parameters可同时返回可学习的参数及名称。

```python
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())
```

```bash
conv1.weight : torch.Size([6, 1, 3, 3])
conv1.bias : torch.Size([6])
fc1.weight : torch.Size([10, 1350])
fc1.bias : torch.Size([10])
```

```python
input = torch.randn(1, 1, 32, 32) # 这里的对应前面fforward的输入是32
out = net(input)
out.size()
torch.Size([1, 1, 32, 32])
torch.Size([1, 6, 30, 30])
torch.Size([1, 6, 15, 15])
torch.Size([1, 1350])
```

```bash
torch.Size([1, 10])
```

```bash
input.size()
```

```bash
torch.Size([1, 1, 32, 32])
```

在反向传播前，先要将所有参数的梯度清零

```python
net.zero_grad() 
out.backward(torch.ones(1,10)) # 反向传播的实现是PyTorch自动实现的，我们只要调用这个函数即可
```

**注意**:torch.nn只支持mini-batches，不支持一次只输入一个样本，即一次必须是一个batch。

也就是说，就算我们输入一个样本，也会对样本进行分批，所以，所有的输入都会增加一个维度，我们对比下刚才的input，nn中定义为3维，但是我们人工创建时多增加了一个维度，变为了4维，最前面的1即为batch-size

我感觉我需要去补一下coursera，而不是在这里看代码

#### 前向传播，后向传播，神经网络

翻出来了[这个](https://www.techbrood.com/zh/news/ai/%E7%94%A8%E4%BA%BA%E8%AF%9D%E8%A7%A3%E9%87%8A%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E9%87%8C%E9%9D%A2%E7%9A%84%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%92%8C%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.html)

![1.png](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/202112311621806.png)

这是一个没有隐藏层的神经网络，左侧输入，右侧输出，前向就是从输入到输出，反向就是从后面获取的Loss传到前面的那个输出层里（比如logistic）

![1552807616619421](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/202112311632948.png)

这是一个带有隐藏层的神经网络，像之前没有隐藏层的时候，我们改一个东西的参数就可以了，现在我们还要改小女孩的参数，所以要用链式求导梯度下降。

这些中间层和output就是传说中的[激活函数](https://zh.wikipedia.org/wiki/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0)，如果不用激活函数，每一层输出都是上一层输入的线性函数，没法拟合非线性函数。

输入数据乘以权重（weight）加上一个偏置（bias），然后应用激活函数得到该神经元的输出，再将此输出传给下一层的神经元。这些weight和bias，再加上L1 L2 batch size之类的 就是超参数。

### 损失函数

## 损失函数

在nn中PyTorch还预制了常用的损失函数，下面我们用MSELoss用来计算均方误差

```python
y = torch.arange(0,10).view(1,10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)
#loss是个scalar，我们可以直接用item获取到他的python类型的数值
print(loss.item()) 
```

28.92203712463379

## 优化器

## 优化器

在反向传播计算完所有参数的梯度后，还需要使用优化方法来更新网络的权重和参数，例如随机梯度下降法(SGD)的更新策略如下：

weight = weight - learning_rate * gradient

在torch.optim中实现大多数的优化方法，例如RMSProp、Adam、SGD等，下面我们使用SGD做个简单的样例

In [10]:

```python
import torch.optim
```

```python
out = net(input) # 这里调用的时候会打印出我们在forword函数中打印的x的大小
criterion = nn.MSELoss()
loss = criterion(out, y)
#新建一个优化器，SGD只需要要调整的参数和学习率
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad() 
loss.backward()

#更新参数
optimizer.step()
```

```bash
torch.Size([1, 1, 32, 32])
torch.Size([1, 6, 30, 30])
torch.Size([1, 6, 15, 15])
torch.Size([1, 1350])
```

这样，神经网络的数据的一个完整的传播就已经通过PyTorch实现了。



后面就没有再细看了

## Fine tuning

这个目前恰好切中自己的需求。

针对于某个任务，自己的训练数据不多，那怎么办？ 没关系，我们先找到一个同类的别人训练好的模型，把别人现成的训练好了的模型拿过来，换成自己的数据，调整一下参数，再训练一遍，这就是微调（fine-tune）。 PyTorch里面提供的经典的网络模型都是官方通过Imagenet的数据集与训练好的数据，如果我们的数据训练数据不够，这些数据是可以作为基础模型来使用的。

1. 新数据集和原始数据集合类似，那么直接可以微调一个最后的FC层或者重新指定一个新的分类器
2. 新数据集比较小和原始数据集合差异性比较大，那么可以使用从模型的中部开始训练，只对最后几层进行fine-tuning
3. 新数据集比较小和原始数据集合差异性比较大，如果上面方法还是不行的化那么最好是重新训练，只将预训练的模型作为一个新模型初始化的数据
4. 新数据集的大小一定要与原始数据集相同，比如CNN中输入的图片大小一定要相同，才不会报错
5. 如果数据集大小不同的话，可以在最后的fc层之前添加卷积或者pool层，使得最后的输出与fc层一致，但这样会导致准确度大幅下降，所以不建议这样做
6. 对于不同的层可以设置不同的学习率，一般情况下建议，对于使用的原始数据做初始化的层设置的学习率要小于（一般可设置小于10倍）初始化的学习率，这样保证对于已经初始化的数据不会扭曲的过快，而使用初始化学习率的新层可以快速的收敛。

更具体的可以看[这里](https://github.com/zergtant/pytorch-handbook/blob/master/chapter4/4.1-fine-tuning.ipynb)

还有一些[可视化](https://github.com/zergtant/pytorch-handbook/blob/master/chapter4/4.2.2-tensorboardx.ipynb)的内容，后面用到了再补充

