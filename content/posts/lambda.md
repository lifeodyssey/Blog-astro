---
title: Some Usefull python function
copyright: true
tags:
  - Research
  - python
categories: 学习笔记
mathjax: true
abbrlink: d39dff82
date: 2020-08-07 21:43:14
---

# Python Lambda

在读别人代码的时候看到的，发现自己对这个东西不是很熟悉，复习一下记个笔记。

**lambda 函数是一种小的匿名函数。**

**lambda 函数可接受任意数量的参数，但只能有一个表达式。**

<!-- more -->

## 语法

`lambda arguments : expression`

执行表达式并返回结果

## 实例

```python
x = lambda a : a + 10
print(x(5))
```

这一句话就定义了一个lambda函数，a是这个函数的参数，a+10是这个函数的表达式，x是这个函数的名字。

Lambda可以接受任意数量的参数，比如

```python
x = lambda a, b, c : a + b + c
print(x(5, 6, 2))
```

这个函数就是三个参数

## 函数内匿名函数

假设我定义了这么一个函数

```python
def myfunc(n):
  return lambda a : a * n
```

这个函数的作用是把a变成n倍。

```python
def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)
```

这样就可以很快速地构建出来这样的一个函数：

```python
def mydoubler(a):
  return n*2
```

不需要想用别的的时候再去定义，比如我还想再来一个三倍的函数，就直接：

```python
mytripler = myfunc(3)
```

# zip

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。

我们可以使用 list() 转换来输出列表。

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 ***** 号操作符，可以将元组解压为列表。

## 语法

```python
zip([iterable, ...])
```

## 实例

```python
>>>a = [1,2,3] 
>>> b = [4,5,6] 
>>> c = [4,5,6,7,8] 
>>> zipped = zip(a,b)     # 返回一个对象 
>>> zipped 
<zip object at 0x103abc288> 
>>> list(zipped)  # list() 转换为列表 
[(1, 4), (2, 5), (3, 6)] 
>>> list(zip(a,c))# 元素个数与最短的列表一致 
[(1, 4), (2, 5), (3, 6)]  
>>> a1, a2 = zip(*zip(a,b))# 与 zip 相反，zip(*) 可理解为解压，返回二维矩阵式 
>>> list(a1) 
[1, 2, 3] 
>>> list(a2) 
[4, 5, 6]
```



# map

**map()** 会根据提供的函数对指定序列做映射。

第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表

## 语法

```python
map(function, iterable, ...)
```

## 实例

```python
>>>def square(x) :            # 计算平方数     
return x ** 2  
>>> map(square, [1,2,3,4,5])   # 计算列表各个元素的平方 
[1, 4, 9, 16, 25] 
>>> map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # 使用 lambda 匿名函数 
[1, 4, 9, 16, 25]  # 提供了两个列表，对相同位置的列表数据进行相加 
>>> map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]) 
[3, 7, 11, 15, 19]
```



# dict

字典是另一种可变容器模型，且可存储任意类型对象。

字典的每个键值 **key=>value** 对用冒号 **:** 分割，每个键值对之间用逗号 **,** 分割，整个字典包括在花括号 **{}** 中 ,格式如下所示：

`'d = {key1 : value1, key2 : value2 }'`

