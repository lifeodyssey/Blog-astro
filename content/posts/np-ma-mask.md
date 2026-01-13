---
title: np.ma.mask
tags:
  - Research
  - python
categories: 学习笔记
mathjax: true
abbrlink: '39e89397'
date: 2020-11-10 10:08:30
copyright:
---

自己前几天一直在跟mask斗争，中文网站上也没什么好的资料，就索性自己整理一下。

<!-- more -->

# 创建

官方提供了三种方法来创建mask，第一种是直接指定mask的位置


```python
import numpy as np
import numpy.ma as ma
```


```python
y = ma.array([1, 2, 3], mask = [0, 1, 0])
y
```




    masked_array(data=[1, --, 3],
                 mask=[False,  True, False],
           fill_value=999999)



第二种是用ma.MaskedArray类，这个不是很常用

x = MaskedArray(data, mask=nomask, dtype=None, copy=False, subok=True,
                ndmin=0, fill_value=None, keep_mask=True, hard_mask=None,
                shrink=True, order=None)


```python
data = np.arange(6).reshape((2, 3))
np.ma.MaskedArray(data, mask=[[False, True, False],
                              [False, False, True]])
```




    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)



这种方法可以直接给整个数组上mask


```python
np.ma.MaskedArray(data, mask=False)
```




```python
masked_array(
  data=[[0, 1, 2],
        [3, 4, 5]],
  mask=[[False, False, False],
        [False, False, False]],
  fill_value=999999)
```




```python
np.ma.MaskedArray(data, mask=True)
```




```python
masked_array(
  data=[[--, --, --],
        [--, --, --]],
  mask=[[ True,  True,  True],
        [ True,  True,  True]],
  fill_value=999999,
  dtype=int64)
```



官方给出的第三种原文和事例如下，但是我始终没有特别理解

A third option is to take the view of an existing array. In that case, the mask of the view is set to nomask if the array has no named fields, or an array of boolean with the same structure as the array otherwise.


```python
x = np.array([1, 2, 3])
```


```python
x.view(ma.MaskedArray)
```




```python
masked_array(data=[1, 2, 3],
             mask=False,
       fill_value=999999)
```




```python
x = np.array([(1, 1.), (2, 2.)], dtype=[('a',int), ('b', float)])
```


```python
x.view(ma.MaskedArray)
```




```python
masked_array(data=[(1, 1.0), (2, 2.0)],
             mask=[(False, False), (False, False)],
       fill_value=(999999, 1.e+20),
            dtype=[('a', '<i8'), ('b', '<f8')])
```



对于我来说我的需求基本就是构造一个和已有mask array相同的数组，对我来说可以用以下方法得到


```python
ma.array(np.zeros_like(y),mask=y.mask)
```




```python
masked_array(data=[0, --, 0],
             mask=[False,  True, False],
       fill_value=999999)
```



除此之外，还有一些函数可以完成效果，比较常用的是这几个

masked_equal(x, value[, copy])

masked_greater(x, value[, copy])

masked_greater_equal(x, value[, copy])	

masked_invalid(a[, copy])

masked_less(x, value[, copy])

masked_less_equal(x, value[, copy])

masked_not_equal(x, value[, copy])

masked_where(condition, a[, copy])

这里masked_where()和masked_invalid()最好用


```python
a = np.arange(4)
ma.masked_where(a <= 2, a)
```




```python
masked_array(data=[--, --, --, 3],
             mask=[ True,  True,  True, False],
       fill_value=999999)
```




```python
a=np.asarray([1,2,3,np.inf,np.nan])
ma.masked_invalid(a)
```




```python
masked_array(data=[1.0, 2.0, 3.0, --, --],
             mask=[False, False, False,  True,  True],
       fill_value=1e+20)
```



在创建里面还有一个事情是比较重要的，那就是fill_value

在创建的时候可以指定fill_value，除此之外，可以通过这个函数来查看fill_value

```python
x = np.ma.array([0, 1.], fill_value=-np.inf)
>>> x.fill_value
-inf
>>> x.fill_value = np.pi
>>> x.fill_value
3.1415926535897931 # may vary
```

对于已经设置好fill_value的，可以用这个函数来改变fill_value

```python
import numpy.ma as ma
>>> a = np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> a = ma.masked_where(a < 3, a)
>>> a
masked_array(data=[--, --, --, 3, 4],
             mask=[ True,  True,  True, False, False],
       fill_value=999999)
>>> ma.set_fill_value(a, -999)
>>> a
masked_array(data=[--, --, --, 3, 4],
             mask=[ True,  True,  True, False, False],
       fill_value=-999)
```

 Nothing happens if a is a not masked array

```python
a = list(range(5))
>>> a
[0, 1, 2, 3, 4]
>>> ma.set_fill_value(a, 100)
>>> a
[0, 1, 2, 3, 4]
>>> a = np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> ma.set_fill_value(a, 100)
>>> a
array([0, 1, 2, 3, 4])
```

在创建完之后，大部分运算都只会针对没有被mask的地方。一些函数可能会出问题，所以像min/max/average这种函数推荐使用np里的函数而不是python内置函数

# 索引

如果索引的地方没有被mask的话，返回值和普通的array一样。

如果是被masked，返回值是masked。

```python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x[0]
1
>>> x[-1]
masked
>>> x[-1] is ma.masked
True
```

# 取消

常用的是这个

```python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x
masked_array(data=[1, 2, --],
             mask=[False, False,  True],
       fill_value=999999)
>>> x.mask = ma.nomask
>>> x
masked_array(data=[1, 2, 3],
             mask=[False, False, False],
       fill_value=999999)
```

这样就可以获取被mask地方的值了，但是要注意的是其实这个返回的array还是带mask的，就是mask全是False罢了，想让他完全变成nomask可以用

```python
x.data                                                                                                              
Out[6]: array([1, 2, 3])
```

这个 返回值就是只有array



# 参考

https://numpy.org/doc/stable/reference/maskedarray.generic.html

https://www.numpy.org.cn/reference/arrays/maskedarray.html

