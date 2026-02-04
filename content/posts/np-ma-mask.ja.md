---
title: np.ma.mask
tags:
  - Research
  - python
categories: 学習ノート
mathjax: true
abbrlink: '39e89397'
date: 2020-11-10 10:08:30
copyright:
lang: ja
---

ここ数日間ずっとmaskと格闘していて、中国語のサイトにも良い資料がなかったので、自分で整理することにしました。

<!-- more -->

# 作成

公式ドキュメントではmaskを作成する3つの方法が提供されています。最初の方法はmaskの位置を直接指定することです：

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

2番目の方法はma.MaskedArrayクラスを使用することですが、あまり一般的ではありません：

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

この方法で配列全体に直接maskをかけることができます：

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

公式ドキュメントの3番目の方法は以下の通りですが、完全には理解できませんでした：

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

私の用途では、基本的に既存のmasked arrayと同じmaskを持つ配列を構築したいです。これは以下のように実現できます：

```python
ma.array(np.zeros_like(y),mask=y.mask)
```

```python
masked_array(data=[0, --, 0],
             mask=[False,  True, False],
       fill_value=999999)
```

また、同様の効果を達成できる関数もあります。よく使われるものは：

masked_equal(x, value[, copy])

masked_greater(x, value[, copy])

masked_greater_equal(x, value[, copy])

masked_invalid(a[, copy])

masked_less(x, value[, copy])

masked_less_equal(x, value[, copy])

masked_not_equal(x, value[, copy])

masked_where(condition, a[, copy])

ここでmasked_where()とmasked_invalid()が最も便利です：

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

作成時にもう一つ重要なのはfill_valueです。

作成時にfill_valueを指定できます。また、この関数でfill_valueを確認できます：

```python
x = np.ma.array([0, 1.], fill_value=-np.inf)
>>> x.fill_value
-inf
>>> x.fill_value = np.pi
>>> x.fill_value
3.1415926535897931 # may vary
```

fill_valueが既に設定されている配列に対して、この関数でfill_valueを変更できます：

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

masked arrayでない場合は何も起こりません：

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

作成後、ほとんどの操作はmaskされていない位置にのみ適用されます。一部の関数には問題がある場合があるので、min/max/averageなどの関数については、Pythonの組み込み関数ではなくnumpyの関数を使用することをお勧めします。

# インデックス

インデックスされた位置がmaskされていない場合、戻り値は通常の配列と同じです。

maskされている場合、戻り値はmaskedになります。

```python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x[0]
1
>>> x[-1]
masked
>>> x[-1] is ma.masked
True
```

# マスクの削除

よく使われる方法は：

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

この方法でmaskされた位置の値を取得できますが、返される配列にはまだmaskがあり、すべてFalseになっているだけです。完全にnomaskにするには：

```python
x.data
Out[6]: array([1, 2, 3])
```

この戻り値は単なる配列です。

# 参考文献

https://numpy.org/doc/stable/reference/maskedarray.generic.html

https://www.numpy.org.cn/reference/arrays/maskedarray.html
