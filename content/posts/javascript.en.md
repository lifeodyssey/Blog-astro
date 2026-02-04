---
title: JavaScript
tags:
  - Javascript
  - 'Software Engineering'
categories: Study Notes
abbrlink: 84ea90f7
password: GTB2o22
date: 2022-06-27 15:45:10
mathjax:
copyright:
lang: en
---

Still need to learn TypeScript later.

<!-- more -->

# Basic

JS does automatic type coercion, so 3/4=0.75

`==` performs type conversion during comparison.
`===` (strict equal) does not perform type conversion.

`var` is function/global scoped, `let` and `const` are block scoped.

## Array Methods

`map()` creates a new array with results of calling a function on every element.

`reduce()` executes a reducer function on each element, accumulating to a single value.

## About `this`

`this` is dynamically bound to the parent context. If extracted from an object, it becomes undefined.
