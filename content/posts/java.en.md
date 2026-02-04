---
title: Java
tags:
  - Java
  - 'Software Engineering'
categories: Study Notes
abbrlink: 7f1ae6d2
date: 2022-04-11 17:49:43
mathjax:
copyright:
password: GTB2o22
lang: en
---

Why not use Go? I'll have to learn Kotlin later anyway.

<!-- more -->

# Installation and Environment Setup

```powershell
winget install Oracle.JDK.17 --location [Installation Destination]
```

Windows command line option.

You can also download OpenJDK from Tsinghua mirror.

## Setting Environment Variables

- After installing JDK, set JAVA_HOME environment variable pointing to the JDK installation directory.
- Windows setup: Find JDK directory like `C:\Program Files\Java\jdk-17`, create JAVA_HOME variable with this path.
- Add bin directory to PATH: `Path=%JAVA_HOME%\bin;<existing paths>`
- Verify with `java -version`

## IntelliJ IDEA

IntelliJ IDEA tutorial: https://www.youtube.com/watch?v=yefmcX57Eyg

Common shortcuts: https://blog.jetbrains.com/idea/2020/03/top-15-intellij-idea-shortcuts/

# Java Basics

Object-oriented programming that I've been avoiding is here again.

Ended up watching Heima tutorials.

# Object-Oriented

## Inheritance

- `extends` is for extending a class
- `implements` is for implementing an interface

## Override vs Overload

Override: Subclass rewrites parent's method, same signature
Overload: Same method name, different parameters

# Cross-Origin Issues

Cross-origin problem is caused by browser's same-origin policy. Same origin means: same protocol + domain + port.
