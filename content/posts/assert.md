---
title: assert
tags:
  - python
categories: 学习笔记
mathjax: true
abbrlink: b1ef4fab
copyright: true
date: 2020-08-18 21:32:57
---

最近在读别人代码的时候发现了好多自己当时学python的时候没有学过的东西

<!-- more -->

# 语法

Python assert用于判断一个表达式，在表达式条件为false的时候触发异常。

语法格式如下：

```python
assert expression
```

等价于

```python
if not expression:
    raise AssertionError
```

后面也可以跟参数:

```python
assert expression [, arguments]
```

等价于：

```python
assert expression [, arguments]
```

# 使用实例

> 检查 先验条件 使用assert，检查 后验条件 使用 异常处理



举个例子来说明一下，在开发中我们经常会遇到读取本地文件的场景。我们定义一个read_file方法。

```python
def read_file(path):
    assert isinstance(file_path, str)
    ...
```

read_file函数要求在开始执行的时候满足一定条件：file_path必须是str类型，这个条件就是***先验条件\***，如果不满足，就不能调用这个函数，如果真的出现了不满足条件的情况，证明代码中出现了bug，这时候我们就可以使用assert语句来对file_path的类型进行推断，提醒程序员修改代码，也可以使用if...raise...语句来实现assert，但是要繁琐很多。在很多优秀的Python项目中都会看到使用assert进行先验判断的情况，平时可以多多留意。

read_file函数在被调用执行后，依然需要满足一定条件，比如file_path所指定的文件需要是存在的，并且当前用户有权限读取该文件，这些条件称为后验条件，对于后验条件的检查，我们需要使用异常来处理。

```python
def read_file(file_path):
    assert isinstance(file_path, str)
    if not check_exist(file_path):
        raise FileNotFoundError()
    if not has_privilege(file_path):
        raise PermissionError()
```

文件不存在和没有权限，这两种情况并不属于代码bug，是代码逻辑的一部分，上层代码捕获异常后可能会执行其他逻辑，因此我们不能接受这部分代码在生产环境中被忽略，这属于***后验条件\***。并且，相比于assert语句只能抛出AssertionError，使用异常可以抛出更详细的错误，方便上层代码针对不同错误执行不同的逻辑。

再比如

```python
import sys
assert ('linux' **in** sys.platform), "该代码只能在 Linux 下执行"

```

# 其他异常处理

顺手补一下其他异常处理

## try...except

这个语句主要是用来可能发生错误的语句里面，比如在跑一个循环的时候有的除以零，有的没有除以零，可以使用try except把除以零的赋值nan避免这个循环停止，让程序继续跑下去。

## try...finally

```python
# try..finally模式
try:
	<statement>        #运行别的代码
finally:
	<statement>        #不管有无异常都会执行 
```

try..finally模式是:

1. 没有异常就先运行try所有语句,再运行finally所有语句.
2. 要是有异常,try执行到异常就跳到finally,然后直接跳出将异常递交给上层的try.控制流并不通过所有try语句.
3. finally能跟在except/else后,优先先执行except/else再执行finally.

由此可知, try…finally 模式更适合于嵌套在try..except内作为保证某些代码一定执行.因为try..except…else要是执行了except就不会执行else,无法保证某个代码必须执行.所以常见的整合模式为:

```python
# 两种模式的嵌套和结合
try:
	<statement1>        #运行测试代码1
	try:
		<statement2>        #运行测试代码2
	finally:
		<statement3>        #不管测试代码2有无异常都会执行
except <name>：
	<statement>        #测试代码1或2发生错误而被捕获,就会执行异常
else:
	<statement>        #测试代码1和2都没有发生错误就会执行
finally:
	<statement4>        #无论两个try有无异常,都会运行一次.
```

PS: 要是finally在except/else前面肯定会报错.因为try后直接给finally,然后会交给上层try.但没有上层try…

实例:

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

### 与try一起使用

raise语句可以很好地用于抛出某个异常从而被try捕获. 更常用于结合if等进行条件检查.例如某变量假定[0,10],<0时抛出一个错,>10抛出另一个错误.

raise一般是`raise exception,args`,args一般采用一个值,来初始化异常类的args属性,也可以直接使用元组来初始化args.

```python
raise <name>    #手工地引发异常
raise <name>,<data>    #传递一个附加的数据(一个值或者一个元组),要是不指定参数,则为None.
raise Exception(data)    #和上面等效.
raise [Exception [, args [, traceback]]]  # 第三个参数是用于跟踪异常对象,基本不用.

try:
	if (i>10):
		raise TypeError(i)
	elif (i<0):
		raise ValueError,i
#下面的e实际是返回错误的对象实例.
except TypeError,e:
	print str(e)+" for i is larger than 10!"
except ValueError,e:
	print str(e)+" for i is less than 0!"
else:
	print "i is between 0 and 10~"
```

# 参考资料

https://stackoverflow.com/questions/40182944/difference-between-raise-try-and-assert

https://www.cnblogs.com/lsdb/p/11063568.html

https://blog.csdn.net/qq_29287973/article/details/78053764

https://www.runoob.com/python3/python3-assert.html

https://zhuanlan.zhihu.com/p/91853234