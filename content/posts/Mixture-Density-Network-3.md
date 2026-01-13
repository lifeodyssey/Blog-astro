---
title: Mixture Density Network 3
tags:
  - 机器学习
  - Inversion
  - Deep Learning
categories:
  - 学习笔记
abbrlink: b6e8cccb
date: 2022-02-22 14:17:57
mathjax:
copyright:
---

继续来看人家写代码的的一些技巧

<!-- more -->

# argparse

[`argparse`](https://docs.python.org/zh-cn/3/library/argparse.html#module-argparse) 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 [`argparse`](https://docs.python.org/zh-cn/3/library/argparse.html#module-argparse) 将弄清如何从 [`sys.argv`](https://docs.python.org/zh-cn/3/library/sys.html#sys.argv) 解析出那些参数。 [`argparse`](https://docs.python.org/zh-cn/3/library/argparse.html#module-argparse) 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息

## 示例

以下代码是一个 Python 程序，它获取一个整数列表并计算总和或者最大值：

```python
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
```

假设上面的 Python 代码保存在名为 `prog.py` 的文件中，它可以在命令行运行并提供有用的帮助信息：

```python
$ python prog.py -h
usage: prog.py [-h] [--sum] N [N ...]

Process some integers.

positional arguments:
 N           an integer for the accumulator

options:
 -h, --help  show this help message and exit
 --sum       sum the integers (default: find the max)
```

当使用适当的参数运行时，它会输出命令行传入整数的总和或者最大值：

```python
$ python prog.py 1 2 3 4
4

$ python prog.py 1 2 3 4 --sum
10
```

如果传入无效参数，则会报出错误：

```python
$ python prog.py a b c
usage: prog.py [-h] [--sum] N [N ...]
prog.py: error: argument N: invalid int value: 'a'
```

这个会自动创建对象

我之前已经搞成了dict 这里重新改成object 我用的是这个

```python
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
       args = Struct(**param)
```
