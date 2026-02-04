---
title: assert
tags:
  - python
categories: Learning Notes
mathjax: true
abbrlink: b1ef4fab
copyright: true
date: 2020-08-18 21:32:57
lang: en
---

Recently, while reading other people's code, I discovered many things I didn't learn when I was studying Python.

<!-- more -->

# Syntax

Python assert is used to evaluate an expression and triggers an exception when the expression condition is false.

The syntax format is as follows:

```python
assert expression
```

Equivalent to:

```python
if not expression:
    raise AssertionError
```

You can also add arguments after it:

```python
assert expression [, arguments]
```

Equivalent to:

```python
assert expression [, arguments]
```

# Usage Examples

> Use assert to check **preconditions**, use exception handling to check **postconditions**

Let me give an example. In development, we often encounter scenarios where we need to read local files. We define a read_file method.

```python
def read_file(path):
    assert isinstance(file_path, str)
    ...
```

The read_file function requires certain conditions to be met before execution: file_path must be of str type. This condition is called a ***precondition***. If not satisfied, this function should not be called. If such a situation does occur, it indicates a bug in the code. At this point, we can use the assert statement to check the type of file_path and remind the programmer to fix the code. You could also use if...raise... statements to achieve the same effect as assert, but it would be much more cumbersome. In many excellent Python projects, you'll see assert being used for precondition checking, so pay attention to this in your daily reading.

After the read_file function is called and executed, certain conditions still need to be met. For example, the file specified by file_path needs to exist, and the current user needs to have permission to read the file. These conditions are called postconditions. For checking postconditions, we need to use exception handling.

```python
def read_file(file_path):
    assert isinstance(file_path, str)
    if not check_exist(file_path):
        raise FileNotFoundError()
    if not has_privilege(file_path):
        raise PermissionError()
```

File not existing and lacking permission - these two situations are not code bugs but part of the code logic. The upper-level code may execute other logic after catching the exception, so we cannot accept this part of the code being ignored in production. This is a ***postcondition***. Additionally, compared to assert statements which can only throw AssertionError, using exceptions allows throwing more detailed errors, making it convenient for upper-level code to execute different logic for different errors.

Another example:

```python
import sys
assert ('linux' in sys.platform), "This code can only run on Linux"
```

# Other Exception Handling

Let me also cover other exception handling methods.

## try...except

This statement is mainly used in code that might produce errors. For example, when running a loop where some iterations divide by zero and others don't, you can use try except to assign nan to the division-by-zero cases to prevent the loop from stopping, allowing the program to continue running.

## try...finally

```python
# try..finally pattern
try:
	<statement>        # run other code
finally:
	<statement>        # will execute regardless of whether there's an exception
```

The try..finally pattern works as follows:

1. If there's no exception, it first runs all try statements, then runs all finally statements.
2. If there's an exception, try executes until the exception, then jumps to finally, and then directly exits to pass the exception to the upper-level try. Control flow does not pass through all try statements.
3. finally can follow except/else, with except/else executing first, then finally.

From this, we can see that the try…finally pattern is more suitable for nesting within try..except to ensure certain code always executes. Because with try..except…else, if except executes, else won't execute, so there's no guarantee that certain code must execute. Therefore, the common integrated pattern is:

```python
# Nesting and combining the two patterns
try:
	<statement1>        # run test code 1
	try:
		<statement2>        # run test code 2
	finally:
		<statement3>        # will execute regardless of whether test code 2 has an exception
except <name>:
	<statement>        # executes if test code 1 or 2 throws a caught error
else:
	<statement>        # executes if neither test code 1 nor 2 throws an error
finally:
	<statement4>        # will run once regardless of whether either try has an exception
```

PS: If finally comes before except/else, it will definitely cause an error. Because after try goes directly to finally, it will then pass to the upper-level try. But there's no upper-level try…

Example:

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

### Using with try

The raise statement can be used effectively to throw an exception to be caught by try. It's more commonly used in combination with if for conditional checking. For example, if a variable is assumed to be in [0,10], throw one error when <0, and throw another error when >10.

raise is generally `raise exception, args`, where args is usually a single value to initialize the exception class's args attribute, or you can directly use a tuple to initialize args.

```python
raise <name>    # manually raise an exception
raise <name>,<data>    # pass additional data (a value or tuple), if no parameter specified, it's None
raise Exception(data)    # equivalent to above
raise [Exception [, args [, traceback]]]  # third parameter is for tracking exception object, rarely used

try:
	if (i>10):
		raise TypeError(i)
	elif (i<0):
		raise ValueError,i
# e below is actually the returned error object instance
except TypeError,e:
	print str(e)+" for i is larger than 10!"
except ValueError,e:
	print str(e)+" for i is less than 0!"
else:
	print "i is between 0 and 10~"
```

# References

https://stackoverflow.com/questions/40182944/difference-between-raise-try-and-assert

https://www.cnblogs.com/lsdb/p/11063568.html

https://blog.csdn.net/qq_29287973/article/details/78053764

https://www.runoob.com/python3/python3-assert.html

https://zhuanlan.zhihu.com/p/91853234

