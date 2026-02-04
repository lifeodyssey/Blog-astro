---
title: np.ma.mask
tags:
  - Research
  - python
categories: Learning Notes
mathjax: true
abbrlink: '39e89397'
date: 2020-11-10 10:08:30
copyright:
lang: en
---

I've been struggling with masks for the past few days, and there wasn't much good material on Chinese websites, so I decided to organize it myself.

<!-- more -->

# Creating

The official documentation provides three methods to create masks. The first is to directly specify the mask positions:

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

The second method uses the ma.MaskedArray class, which is not very commonly used:

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

This method can directly mask the entire array:

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

The third method from the official documentation is as follows, but I never fully understood it:

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

For my needs, I basically want to construct an array with the same mask as an existing masked array. I can achieve this with:

```python
ma.array(np.zeros_like(y),mask=y.mask)
```

```python
masked_array(data=[0, --, 0],
             mask=[False,  True, False],
       fill_value=999999)
```

Additionally, there are some functions that can achieve similar effects. The commonly used ones are:

masked_equal(x, value[, copy])

masked_greater(x, value[, copy])

masked_greater_equal(x, value[, copy])

masked_invalid(a[, copy])

masked_less(x, value[, copy])

masked_less_equal(x, value[, copy])

masked_not_equal(x, value[, copy])

masked_where(condition, a[, copy])

Here masked_where() and masked_invalid() are the most useful:

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

Another important thing in creation is fill_value.

You can specify fill_value when creating. Additionally, you can view fill_value with this function:

```python
x = np.ma.array([0, 1.], fill_value=-np.inf)
>>> x.fill_value
-inf
>>> x.fill_value = np.pi
>>> x.fill_value
3.1415926535897931 # may vary
```

For arrays with fill_value already set, you can use this function to change fill_value:

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

Nothing happens if a is not a masked array:

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

After creation, most operations only apply to non-masked positions. Some functions may have issues, so for functions like min/max/average, it's recommended to use numpy functions rather than Python built-in functions.

# Indexing

If the indexed position is not masked, the return value is the same as a regular array.

If it is masked, the return value is masked.

```python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x[0]
1
>>> x[-1]
masked
>>> x[-1] is ma.masked
True
```

# Removing Masks

The commonly used method is:

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

This way you can get the values at masked positions, but note that the returned array still has a mask, just all False. To make it completely nomask, you can use:

```python
x.data
Out[6]: array([1, 2, 3])
```

This return value is just the array.

# References

https://numpy.org/doc/stable/reference/maskedarray.generic.html

https://www.numpy.org.cn/reference/arrays/maskedarray.html
