---
title: Mixture Density Network 3
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
categories:
  - 学習ノート
abbrlink: b6e8cccb
slug: mixture-density-network-3
date: 2022-02-22 14:17:57
mathjax:
copyright:
lang: ja
---

コードで使われているいくつかのテクニックを引き続き見ていきましょう。

<!-- more -->

# argparse

[`argparse`](https://docs.python.org/ja/3/library/argparse.html#module-argparse)モジュールを使うと、ユーザーフレンドリーなコマンドラインインターフェースを簡単に作成できます。プログラムが必要とする引数を定義すると、[`argparse`](https://docs.python.org/ja/3/library/argparse.html#module-argparse)が[`sys.argv`](https://docs.python.org/ja/3/library/sys.html#sys.argv)からそれらの引数を解析する方法を判断します。[`argparse`](https://docs.python.org/ja/3/library/argparse.html#module-argparse)モジュールはヘルプと使用方法のメッセージも自動生成し、ユーザーが無効な引数を渡した場合にエラーを発行します。

## 例

以下のコードは、整数のリストを受け取り、合計または最大値を出力するPythonプログラムです：

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

上記のPythonコードが`prog.py`というファイルに保存されていると仮定すると、コマンドラインで実行でき、有用なヘルプメッセージを提供します：

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

適切な引数で実行すると、コマンドライン整数の合計または最大値を出力します：

```python
$ python prog.py 1 2 3 4
4

$ python prog.py 1 2 3 4 --sum
10
```

無効な引数が渡されると、エラーを発行します：

```python
$ python prog.py a b c
usage: prog.py [-h] [--sum] N [N ...]
prog.py: error: argument N: invalid int value: 'a'
```

これは自動的にオブジェクトを作成します。

以前dictに変換していましたが、ここでobjectに戻しています。私が使っているのはこれです：

```python
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
       args = Struct(**param)
```