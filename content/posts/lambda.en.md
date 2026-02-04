---
title: Some Useful Python Functions
copyright: true
tags:
  - Research
  - python
categories: Learning Notes
mathjax: true
abbrlink: d39dff82
date: 2020-08-07 21:43:14
lang: en
---

# Python Lambda

I saw this while reading someone else's code and realized I wasn't very familiar with it, so I'm reviewing and taking notes.

**A lambda function is a small anonymous function.**

**A lambda function can take any number of arguments, but can only have one expression.**

<!-- more -->

## Syntax

`lambda arguments : expression`

Executes the expression and returns the result.

## Example

```python
x = lambda a : a + 10
print(x(5))
```

This single line defines a lambda function, where `a` is the parameter, `a+10` is the expression, and `x` is the function name.

Lambda can accept any number of arguments, for example:

```python
x = lambda a, b, c : a + b + c
print(x(5, 6, 2))
```

This function takes three parameters.

## Anonymous Functions Inside Functions

Suppose I define a function like this:

```python
def myfunc(n):
  return lambda a : a * n
```

This function multiplies `a` by `n`.

```python
def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)
```

This quickly builds a function like:

```python
def mydoubler(a):
  return n*2
```

No need to define another function when you want something else. For example, if I want a tripler function, I can just do:

```python
mytripler = myfunc(3)
```

# zip

The **zip()** function takes iterable objects as arguments, packs corresponding elements from the objects into tuples, and returns an object composed of these tuples. This saves a lot of memory.

We can use list() to convert and output a list.

If the iterators have different numbers of elements, the returned list length matches the shortest object. Using the **\*** operator, you can unzip tuples into lists.

## Syntax

```python
zip([iterable, ...])
```

## Example

```python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # Returns an object
>>> zipped
<zip object at 0x103abc288>
>>> list(zipped)  # list() converts to list
[(1, 4), (2, 5), (3, 6)]
>>> list(zip(a,c))# Number of elements matches shortest list
[(1, 4), (2, 5), (3, 6)]
>>> a1, a2 = zip(*zip(a,b))# Opposite of zip, zip(*) can be understood as unzip
>>> list(a1)
[1, 2, 3]
>>> list(a2)
[4, 5, 6]
```

# map

**map()** applies a provided function to map over a specified sequence.

The first parameter `function` calls the function on each element in the parameter sequence, returning a new list containing the return values of each function call.

## Syntax

```python
map(function, iterable, ...)
```

## Example

```python
>>>def square(x) :            # Calculate square
return x ** 2
>>> map(square, [1,2,3,4,5])   # Calculate square of each element
[1, 4, 9, 16, 25]
>>> map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # Using lambda anonymous function
[1, 4, 9, 16, 25]  # Provided two lists, adds elements at same positions
>>> map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
[3, 7, 11, 15, 19]
```

# dict

A dictionary is another mutable container model that can store objects of any type.

Each key-value pair **key=>value** in a dictionary is separated by a colon **:**, each pair is separated by a comma **,**, and the entire dictionary is enclosed in curly braces **{}**. The format is:

`d = {key1 : value1, key2 : value2}`
