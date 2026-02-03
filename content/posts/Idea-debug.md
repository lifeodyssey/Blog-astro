---
title: Idea debug/Gradle/Building tools
draft: true
date: 2022-05-12 23:12:22 
tags:
  - Java 
  - 'Software Engineering' 
categories: 学习笔记
abbrlink: 984bc8b6
mathjax:
copyright:
password: GTB2o22
---



<!-- more -->

# Idea Debug

00:00 - Intro
1:20 Why Debug?
10:30 Start/stop/restart debugger
14:08 Suspend/resume
15:24 Setting breakpoint
19:00 Disable breakpoints
19:56 Steps
23:45 Frames (bg color)
25:35 Inline debugger
27:03 Variables views
28:00 Quick evaluate
30:35 Evaluate
33:45 Watches
35:50 Set value
38:00 Breakpoint conditions
40:17 Questions
youtube上可以直接跳转分区 https://www.youtube.com/watch?v=59RC8gVPlvk

# Gradle

https://link.springer.com/article/10.1007/s10872-022-00642-9

windows只能手动安装

# Building Tools

这里只讲了gradles

## 啥是构建和构建工具工具

> In **software development**, a build is the process of converting  **source code** files into **standalone software artifact(s)** that can  be run on a computer

就是把.c/.py/.java打包成jar/exe/apk的过程

经过的过程是

.java->.class->.jar

第一个过程叫compile 第二个过程叫package

> Build tools are **software applications** that help in **build  automation**
>
> Build automation is the  **process** of **automating** the  creation of a software build  and the associated processes

├── app ✅                                       子项目目录
│         ├── build ✅                           子项目生成 artifacts 的目录
│         ├── build.gradle ✅                    子项目的 gradle build 脚本
│         └── src        
│             ├── main ✅                        子项目的源代码及配置文件目录
│             └── test ✅                        子项目的测试代码及配置文件目录
├── gradle        
│         └── wrapper        
│             ├── gradle-wrapper.jar ✅          包含用于下载 Gradle 版本代码的 jar 文件
│             └── gradle-wrapper.properties ✅   控制 Gradle Wrapper 运行时行为的属性文件
├── gradlew ✅                                   使用 Gradle Wrapper 在 Linux/Unix 平台构建的执行脚本
├── gradlew.bat ✅                               使用 Gradle Wrapper 在 Windows 平台构建的执行脚本
└── settings.gradle ✅                           项目的设置文件，定义了子项目的位置

Maven、Ant、Gradle 是 Java 技术栈的构建工具。Gulp 是 JavaScript 技术栈的构建工具

下面哪种方式可以修改wrapper中的gradle version

gradle wrapper --gradle-version 6.8.3 【正确答案】

build.gradle中增加 wrapper { gradleVersion = '6.8.3’ } 【正确答案】

手动修改gradle/wrapper/gradle.properties文件中的distributionUrl属性 【正确答案】

哪一个命令可以列出所有的gradle task

./gradlew tasks 【正确答案】

如果 task A 依赖 task B， task B 依赖 task C，Gradle会首先运行 task C

build task的依赖task列表中包含下列哪些task

clean

test 【正确答案】

check 【正确答案】

jar 【正确答案】
