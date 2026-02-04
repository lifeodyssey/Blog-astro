---
title: assert
tags:
  - python
categories: 学習ノート
mathjax: true
abbrlink: b1ef4fab
copyright: true
date: 2020-08-18 21:32:57
lang: ja
---

最近、他の人のコードを読んでいる時に、Pythonを勉強していた時には学ばなかったことをたくさん発見しました。

<!-- more -->

# 構文

Python assertは式を評価し、式の条件がfalseの場合に例外をトリガーするために使用されます。

構文形式は以下の通りです：

```python
assert expression
```

これは以下と同等です：

```python
if not expression:
    raise AssertionError
```

後ろに引数を追加することもできます：

```python
assert expression [, arguments]
```

これは以下と同等です：

```python
assert expression [, arguments]
```

# 使用例

> **事前条件**のチェックにはassertを使用し、**事後条件**のチェックには例外処理を使用します

例を挙げて説明しましょう。開発では、ローカルファイルを読み込むシナリオによく遭遇します。read_fileメソッドを定義します。

```python
def read_file(path):
    assert isinstance(file_path, str)
    ...
```

read_file関数は実行開始時に特定の条件を満たす必要があります：file_pathはstr型でなければなりません。この条件は***事前条件***と呼ばれます。満たされない場合、この関数を呼び出すべきではありません。もしそのような状況が発生した場合、コードにバグがあることを示しています。この時、assert文を使用してfile_pathの型をチェックし、プログラマーにコードを修正するよう促すことができます。if...raise...文を使用してassertと同じ効果を達成することもできますが、はるかに面倒です。多くの優れたPythonプロジェクトでは、事前条件のチェックにassertが使用されているのを見かけますので、日常的に注意してください。

read_file関数が呼び出されて実行された後も、特定の条件を満たす必要があります。例えば、file_pathで指定されたファイルが存在し、現在のユーザーがそのファイルを読み取る権限を持っている必要があります。これらの条件は事後条件と呼ばれます。事後条件のチェックには、例外処理を使用する必要があります。

```python
def read_file(file_path):
    assert isinstance(file_path, str)
    if not check_exist(file_path):
        raise FileNotFoundError()
    if not has_privilege(file_path):
        raise PermissionError()
```

ファイルが存在しない、権限がない - これら2つの状況はコードのバグではなく、コードロジックの一部です。上位コードは例外をキャッチした後、他のロジックを実行する可能性があるため、この部分のコードが本番環境で無視されることは受け入れられません。これは***事後条件***です。また、assert文はAssertionErrorしかスローできないのに対し、例外を使用するとより詳細なエラーをスローでき、上位コードが異なるエラーに対して異なるロジックを実行するのに便利です。

別の例：

```python
import sys
assert ('linux' in sys.platform), "このコードはLinuxでのみ実行できます"
```

# その他の例外処理

ついでに他の例外処理についても補足します。

## try...except

この文は主にエラーが発生する可能性のあるコードで使用されます。例えば、ループを実行する際に、ゼロ除算が発生するものとしないものがある場合、try exceptを使用してゼロ除算のケースにnanを割り当て、ループが停止するのを防ぎ、プログラムを継続して実行させることができます。

## try...finally

```python
# try..finallyパターン
try:
	<statement>        # 他のコードを実行
finally:
	<statement>        # 例外の有無に関わらず実行される
```

try..finallyパターンは以下のように動作します：

1. 例外がない場合、まずすべてのtry文を実行し、次にすべてのfinally文を実行します。
2. 例外がある場合、tryは例外まで実行し、finallyにジャンプし、その後直接終了して上位のtryに例外を渡します。制御フローはすべてのtry文を通過しません。
3. finallyはexcept/elseの後に続くことができ、except/elseが先に実行され、次にfinallyが実行されます。

このことから、try…finallyパターンはtry..except内にネストして、特定のコードが必ず実行されることを保証するのに適していることがわかります。try..except…elseでは、exceptが実行されるとelseは実行されないため、特定のコードが必ず実行されることを保証できません。したがって、一般的な統合パターンは以下の通りです：

```python
# 2つのパターンのネストと組み合わせ
try:
	<statement1>        # テストコード1を実行
	try:
		<statement2>        # テストコード2を実行
	finally:
		<statement3>        # テストコード2の例外の有無に関わらず実行される
except <name>:
	<statement>        # テストコード1または2でエラーがキャッチされた場合に実行
else:
	<statement>        # テストコード1と2の両方でエラーが発生しなかった場合に実行
finally:
	<statement4>        # 両方のtryの例外の有無に関わらず、一度実行される
```

PS: finallyがexcept/elseの前にある場合、必ずエラーになります。tryの後に直接finallyに行き、その後上位のtryに渡されるからです。しかし上位のtryがありません…

例：

```python
#!/usr/bin/python

try:
   fh = open("testfile", "w")
   try:
      fh.write("This is my test file for exception handling!!")
   finally:
      print "Going to close the file"
      fh.close()
except IOError:
   print "Error: can't find file or read data"
```

## raise

### tryと一緒に使用

raise文は、tryでキャッチされる例外をスローするのに効果的に使用できます。条件チェックのためにifと組み合わせて使用されることが多いです。例えば、変数が[0,10]の範囲にあると仮定し、<0の場合は1つのエラーをスロー、>10の場合は別のエラーをスローします。

raiseは一般的に`raise exception, args`の形式で、argsは通常、例外クラスのargs属性を初期化するための単一の値です。タプルを直接使用してargsを初期化することもできます。

```python
raise <name>    # 手動で例外を発生させる
raise <name>,<data>    # 追加データ（値またはタプル）を渡す、パラメータを指定しない場合はNone
raise Exception(data)    # 上記と同等
raise [Exception [, args [, traceback]]]  # 3番目のパラメータは例外オブジェクトの追跡用、ほとんど使用されない

try:
	if (i>10):
		raise TypeError(i)
	elif (i<0):
		raise ValueError,i
# 以下のeは実際には返されたエラーオブジェクトのインスタンス
except TypeError,e:
	print str(e)+" for i is larger than 10!"
except ValueError,e:
	print str(e)+" for i is less than 0!"
else:
	print "i is between 0 and 10~"
```

# 参考資料

https://stackoverflow.com/questions/40182944/difference-between-raise-try-and-assert

https://www.cnblogs.com/lsdb/p/11063568.html

https://blog.csdn.net/qq_29287973/article/details/78053764

https://www.runoob.com/python3/python3-assert.html

https://zhuanlan.zhihu.com/p/91853234


