---
title: Mixture Density Network 3
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
categories:
  - Learning Notes
abbrlink: b6e8cccb
slug: mixture-density-network-3
date: 2022-02-22 14:17:57
mathjax:
copyright:
lang: en
---

Let's continue looking at some coding techniques used in the code.

<!-- more -->

# argparse

The [`argparse`](https://docs.python.org/3/library/argparse.html#module-argparse) module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and [`argparse`](https://docs.python.org/3/library/argparse.html#module-argparse) will figure out how to parse those out of [`sys.argv`](https://docs.python.org/3/library/sys.html#sys.argv). The [`argparse`](https://docs.python.org/3/library/argparse.html#module-argparse) module also automatically generates help and usage messages, and issues errors when users give the program invalid arguments.

## Example

The following code is a Python program that takes a list of integers and produces either the sum or the max:

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

Assuming the above Python code is saved into a file called `prog.py`, it can be run at the command line and provides useful help messages:

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

When run with the appropriate arguments, it prints either the sum or the max of the command-line integers:

```python
$ python prog.py 1 2 3 4
4

$ python prog.py 1 2 3 4 --sum
10
```

If invalid arguments are passed in, it will issue an error:

```python
$ python prog.py a b c
usage: prog.py [-h] [--sum] N [N ...]
prog.py: error: argument N: invalid int value: 'a'
```

This automatically creates objects.

I had previously converted it to a dict, but here I'm changing it back to an object. I'm using this:

```python
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
       args = Struct(**param)
```
