---
title: 便利なPython関数
copyright: true
tags:
  - Research
  - python
categories: 学習ノート
mathjax: true
abbrlink: d39dff82
date: 2020-08-07 21:43:14
lang: ja
---

# Python Lambda

他の人のコードを読んでいるときに見かけて、あまり馴染みがないことに気づいたので、復習してメモを取ります。

**lambda関数は小さな無名関数です。**

**lambda関数は任意の数の引数を受け取ることができますが、式は1つだけです。**

<!-- more -->

## 構文

`lambda arguments : expression`

式を実行して結果を返します。

## 例

```python
x = lambda a : a + 10
print(x(5))
```

この1行でlambda関数を定義しています。`a`はこの関数のパラメータ、`a+10`はこの関数の式、`x`はこの関数の名前です。

Lambdaは任意の数の引数を受け取ることができます。例えば：

```python
x = lambda a, b, c : a + b + c
print(x(5, 6, 2))
```

この関数は3つのパラメータを取ります。

## 関数内の無名関数

このような関数を定義したとします：

```python
def myfunc(n):
  return lambda a : a * n
```

この関数は`a`を`n`倍にします。

```python
def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)
```

これで以下のような関数を素早く構築できます：

```python
def mydoubler(a):
  return n*2
```

別のものが欲しいときに再定義する必要はありません。例えば、3倍の関数が欲しい場合は：

```python
mytripler = myfunc(3)
```

# zip

**zip()** 関数はイテラブルオブジェクトを引数として受け取り、オブジェクト内の対応する要素をタプルにパックし、これらのタプルで構成されるオブジェクトを返します。これによりメモリを節約できます。

list()を使用してリストに変換して出力できます。

各イテレータの要素数が異なる場合、返されるリストの長さは最短のオブジェクトと同じになります。**\*** 演算子を使用すると、タプルをリストに解凍できます。

## 構文

```python
zip([iterable, ...])
```

## 例

```python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # オブジェクトを返す
>>> zipped
<zip object at 0x103abc288>
>>> list(zipped)  # list()でリストに変換
[(1, 4), (2, 5), (3, 6)]
>>> list(zip(a,c))# 要素数は最短のリストと一致
[(1, 4), (2, 5), (3, 6)]
>>> a1, a2 = zip(*zip(a,b))# zipの逆、zip(*)は解凍と理解できる
>>> list(a1)
[1, 2, 3]
>>> list(a2)
[4, 5, 6]
```

# map

**map()** は提供された関数に基づいて指定されたシーケンスをマッピングします。

最初のパラメータ`function`はパラメータシーケンスの各要素に対して関数を呼び出し、各関数呼び出しの戻り値を含む新しいリストを返します。

## 構文

```python
map(function, iterable, ...)
```

## 例

```python
>>>def square(x) :            # 平方を計算
return x ** 2
>>> map(square, [1,2,3,4,5])   # 各要素の平方を計算
[1, 4, 9, 16, 25]
>>> map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # lambda無名関数を使用
[1, 4, 9, 16, 25]  # 2つのリストを提供し、同じ位置の要素を加算
>>> map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
[3, 7, 11, 15, 19]
```

# dict

辞書は任意の型のオブジェクトを格納できる別の可変コンテナモデルです。

辞書の各キーと値のペア **key=>value** はコロン **:** で区切られ、各ペアはカンマ **,** で区切られ、辞書全体は波括弧 **{}** で囲まれます。形式は以下の通りです：

`d = {key1 : value1, key2 : value2}`
