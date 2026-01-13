---
title: TDD
tags: 'Software Engineering'
categories: 学习笔记
password: yuanlainiyewanyuanshen
abbrlink: 715050f6
date: 2022-07-18 16:22:18
mathjax:
copyright:
---

# TDD demo

# Tasking

## Tasking 理论

### Tasking 铁三角

- 有价值

  - 有业务价值

  - 能够实现一个功能
  - 用户能够使用，感觉到软件的变化

- 足够小

  - 让干活的人能够开始干活
  - 而不是"动不了手"或是"瞎干活"
  - 是对个人来说相对的小，也不能太小，不到代码实现层面
  - 跟"价值"平衡

- 说人话
  - 沟通:三天能看的懂
    - 跟非程序员(用户)沟通
    - 跟你的pair沟通
  - 产品思维-call back有价值
    - 用使用场景和解决的问题来思考
    - 而不是for map等技术细节

### 技术细节

- 简化场景

  - 手段
    - 预算砍半法:再砍一半就不能做了，五到十分钟能做完，直接google就能做
    - 极端假设法:写死某一部分输出，只做其中一部分，干掉大部分变化因素再加回来。例如写list的时候写死completed的那部分
    - 一档起步法:

  - 举例
    - 手机能怎么简化->从智能机一直简化到拨号机
    - 微信/QQ怎么简化->从现在简化到只能发文字信息的messager
    - 东方红卫星到天宫空间站

- 降维减量
  - 维度
    - 手段:找名次
    - 举例:查看，添加删除，切换状态三个维度
  - 数量
    - 找数量
    - 举例:add task 时的name，只有一个单词作为name，或者写死name，然后再想怎么读入name
- Happy First
  - 先做happy path
  - 再做sad path

- 高频优先
  - Todo App的List功能肯定比init功能用到的频率更高

### 典型反模式

- 不说人话，全是技术术语，甚至伪代码片段
- 万能tasking，放之四海而皆准。例如输入->处理->输出
- 拆的太大

### 其他注意事项

- 不用全写完，边做边写
- 不用全写对，边做边改
  - 有的时候还要再拆
  - 有的时候还要合并
  - 有的时候还要删除一些错误的预判

### 跟TDD如何结合

每个task对应TDD时的一个或多个tests

### 最后的提醒

- 不要被输入
- 不要为了写一个好的task列表而做task
- 重在效果，不在形式
  - 做正确的是情
  - 能动手能起步
  - 能有助于沟通

## 练习

### Bowling Game

规则如下

- 一场游戏有10个格子
- 每一个格子有10个瓶子，初始可以扔两次球，第10个格子初始可以投三次
- 每一格有全中，补中和失误三种情况
- 全中的情况下，同一个格子再扔两次，分数为这三次的总和
- 补中的情况下，再扔一次，分数为这三次的总和

Tasking

- 计算一个格子
  - 全中
    - 再扔两次
  - 补中
    - 再扔一次
  - 失误
- 计算多个格子
- 计算最后一个格子

最初的第一个任务为，对于一个只扔两次的格子，输出他全中的分数

### Args

- 没有参数
- 参数结构正确
  - 没有l标记
    - 一个标记，没有值
      - 获取缺省值
    - 一个标记，一个值
      - 传入进行处理
    - 多个标记，多个值
      - 传入进行处理
  - 有l标记
- 参数结构错误

# Clean Code

## 关于写代码

> ​	任何傻瓜都能写出来机器可以读懂的代码，优秀的程序员要写的是人可以读懂的代码
>
> 代码主要是写给人看的，偶尔让机器执行一下
>
> 看代码的时间远多于改代码的时间10x
>
> 好的设计是显而易见没有问题，糟糕的涉及是没有显而易见的问题

我们看中的是:

- Readability!Maintainability!
- ✓Problem Solver × Trouble Maker



## 什么是Clean Code

好的代码的总称

## 学习内容

### Clean Code

书 :代码整洁之道

**一些现在就要注意和强制做到的一些点**

1. **1-10-50***（极个别情况可有例外）
   - 每个方法不超过一层缩进
     - try-catch和 JS callback可例外
   - 每个方法最多不超过10行
     - 不包括花括号和名字本身
     - try-catch和fetch API等场景可例外
     - 不要强制把多行写成一行
   - 每个类最多不超过50行
     - import类的语句不算
2. 合理的命名:变量 常量 方法 类 枚举值 文件等等
3. 方法顺序 P29-P32
4. "No" Comments
5. "No\" Else
   - Return Early Pattern

### 4 Rules of Simple Design

![Clean Code — 4 Rules of Simple Design | by Anneke Dwi | The Startup | Medium](https://miro.medium.com/max/879/0*mmKu9oV-Cca0Tuib.png)

### The SOLID Prinnciples

- Single Responsibility
- Open/Closed
- Liskov Substitution
- Interface Segregation
- Dependency Inversion

### Design Patterns

https://refactoring.guru/design-patterns/catalog

![image-20220719164855097](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202207191648155.png)

Coach推荐先红后蓝

## 推荐资源

### 推荐以下学习视频

1. 📺 [Lachhh Clean Code Part-1](https://www.bilibili.com/video/BV1rW41127M5/?spm_id_from=333.788.recommend_more_video.1)

2. 📺 [Lachhh Clean Code Part-2](https://www.bilibili.com/video/BV19W411y7GS/?spm_id_from=333.788.recommend_more_video.-1)

   ![image-20220719171122038](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202207191711099.png)

   

3. 📺 [整洁代码 Clean Code](https://www.bilibili.com/video/BV1tb4y117ee?p=84)

4. 📺 [编写可读代码的艺术](https://www.bilibili.com/video/BV1qs411L7tH?p=1)

5. 📺 [五分钟学设计模式](https://space.bilibili.com/59546029/search/video?keyword=五分钟学设计模式)

### 推荐以下经典书籍 📚

1. Robert C. Martin《代码整洁之道》
   英文名：Clean Code: A Handbook of Agile Software Craftsmanship
2. Robert C. Martin《程序员的职业素养》
   英文名：The Clean Coder：A Code of Conduct for Professional Programmers
3. Robert C. Martin《敏捷软件开发：原则、模式与实践》
   Agile Software Development: Principles, Patterns, and Practices
4. 《高效程序员的 45 个习惯》
5. 《实现模式》
6. 设计模式相关的书
   1. 《设计模式》
   2. 《Head First 设计模式》
   3. 《大话设计模式》
   4. 《重学 Java 设计模式》

推荐阅读顺序：

1. 《实现模式》《代码整洁之道》
2. 《程序员的职业素养》《高效程序员的 45 个习惯》
3. 《敏捷软件开发：原则、模式与实践》
4. 《Head First 设计模式》《设计模式》《大话设计模式》《重学 Java 设计模式》

# TDD

## Refactoring

### Code Smell

1. Duplicate Code
2. Long Method
3. Large Class
4. Long Parameter List
5. Primitive Obsession
6. Data Clumps
7. Switch Statements
8. Feature Envy
9. Comments

### Refactoring Techniques

1. Extract Variable
2. Inline Temp
3. Extract Method
4. Inline Method
5. ……

这里强调一下我们 现在 **就要注意和 强制做到**的一些点

1. 使用快捷键
2. 重构是不能破坏代码的功能，要始终可以编译、运行
3. 不能重构着重构着就开始写上新代码了

## TDD

![Why Test-Driven Development (TDD) | Marsner Technologies](https://marsner.com/wp-content/uploads/test-driven-development-TDD.png)

### Three laws of TDD

![image-20220719231926004](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202207192319086.png)

### 理解 Test Double 里的 Stub 和 Mock

1. 理解为何需要使用 Stub 和 Mock
2. 能够对两者进行区分
3. 可以使用 Mockito 来在测试中实现 Stub 和 Mock
这里强调一下我们 现在 就要注意和 强制做到的一些点
• Test First
• Follow 3 Laws

# 演示笔记

## Note

extend selection

ctrl+D 复制粘贴

ctrl上下快速跳转

先通过测试，这一步代码可以很丑陋，然后再去重构为clean code

有一些思路就写两行代码运行一下测试

新的需要做的事情立刻加到tasking里面

git commit --amend --no-edit将这次的commit合并到上次

让每次测试的开始时环境一致的方法

```java
@BeforeEach 

void setUp
    
```

可以把一个默认的初始值给传入到文件里或者清空文件

```java
@Aftereach
void tearDown()
```

在执行完每个测试之后处理一些东西

Introduce Object Parameter

验证异常也是测试

git stash https://git-scm.com/book/zh/v2/Git-%E5%B7%A5%E5%85%B7-%E8%B4%AE%E8%97%8F%E4%B8%8E%E6%B8%85%E7%90%86 这是干嘛的没看懂

步子太大的时候可以再度taksing

statichttps://www.cnblogs.com/dolphin0520/p/3799052.html

final https://www.cnblogs.com/dolphin0520/p/3736238.html

1. 当步子有点大时该怎么办？不要忘记做 tasking。
2. 模型序列化和反序列化的逻辑要放到一起
3. 要对 `equals` 进行充分的单元测试
4. 基于当前代码的 模块/类 的划分 添加/补充 相应的测试以完成新的功能需求
5. 如何先重构以便让新功能添加更加简单

extract super class

## Test Structure：Given-When-Then、AAA

Nested test class 

将测试附到某个具体的类上https://www.petrikainulainen.net/programming/testing/junit-5-tutorial-writing-nested-tests/

```java
//AAA
//Arrange-Act-Assert
//Given- Arrange
final var app=new App();
//when-Act
final var result=app.run();
//Then-Assert
Assertions.assertEquals(expectedResult,result);
```



## Integration Test vs Unit Test

Integration Test: 读写真实的文件或者文件，几乎是针对整个软件的功能。如果把标准输出也测试了 那就是End-to-End test/e2e test

Unit Test:只针对某一个section进行测试，一般不做泛指

### What is Test

SUT: System Under Test 

### 边界隔离

![image-20220722173921839](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202207221739980.png)

以边界去看输入输出

里面有什么

![image-20220722174014544](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202207221740630.png)

不是所有代码都写在了App class里的

App相当于乐队的指挥，让别的class去干活

![image-20220722174311604](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202207221743719.png)

- 不是所有代码都写在了App class里的
- 越靠近用户界面UI越难做测试
- 有些时候做简化和忽略处理

#### DOC

Depended On Component

![image-20220722193718447](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202207221937494.png)

在测试SUT的时候需要引入DOC

在这种情况下可以洗的一个fake Task Repository

技巧Dependency Injection(DI, do not confuse with dependence inversion principle)

https://stackoverflow.com/questions/46709170/difference-between-dependency-injection-and-dependency-inversion/46745172#46745172

创建一个匿名类 

复写override

**我没看懂**

https://www.youtube.com/watch?v=IKD2-MAkXyQ&t=4s



## Test Double: Mock

mockito frameworkhttps://site.mockito.org/

```java
@BeforeEach
void setUp(){
taskReo=mock(TaskRepo.class)
}
```

在使用了这个之后，就不会跟数据文件产生交互了

equal 方法需要比较hash code，

快捷键cmd+n/alt+fn+f12，否则的话比较的是引用而不是值

Test Double的意思就是测试替身

Stub用于间接输入

## Test Double: Double

order type

在setup时给mock过来的class一个输入
