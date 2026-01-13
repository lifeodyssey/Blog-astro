---
title: java
tags:
  - Java
  - 'Software Engineering' 
categories: 学习笔记
abbrlink: 7f1ae6d2
date: 2022-04-11 17:49:43
mathjax:
copyright:
password: GTB2o22
---

为什么不用go呢 后面还得学kotlin

<!-- more -->

# 安装与环境配置

```powershell
winget install Oracle.JDK.17 --location [Installation Destination] [--accept-package-agreements] [--accept-source-agreements]
```

windows下可以使用的命令行

也可以在清华下载open JDK

https://mirrors.tuna.tsinghua.edu.cn/AdoptOpenJDK/11/jdk/x64/windows/

## 设置环境变量

- 安装完 Java SE（也可称作 JDK，后面会用 JDK 进行表述） 后，需要设置一个名为 JAVA_HOME 的环境变量，它指向 JDK 的安装目录。
- Windows 设置教程
  - 找到 JDK 安装目录，类似：`C:\Program Files\Java\jdk-17`，创建一个名称为 JAVA_HOME 的环境变量，其值为👆找到的安装目录
  - 然后，找到系统环境变量 PATH ，把JAVA_HOME的bin目录附加到上去。加完之后，类似：`Path=%JAVA_HOME%\bin;<现有的其他路径>`
  - 命令行里输入 java -version，返回 java 对应版本信息，证明此步骤已完成
  - 如无法显示，可先尝试重启电脑，再次尝试

## Intellij IDEA

Intellij IDEA 使用教程 ([![img](https://www.youtube.com/s/desktop/f8e3757f/img/favicon_32x32.png)IntelliJ IDEA | Full Course | 2020](https://www.youtube.com/watch?v=yefmcX57Eyg)）

Intellij IDEA 常用快捷键：[文档版](https://blog.jetbrains.com/idea/2020/03/top-15-intellij-idea-shortcuts/)，[视频版 8min](https://www.youtube.com/watch?v=QYO5_riePOQ)

一些你应该知道是用来干嘛的网站https://www.cnblogs.com/gentlescholar/p/15145771.html 

https://zhuanlan.zhihu.com/p/484244128

https://www.exception.site/essay/idea-reset-eval

这个使用教程太长了不想看

# java 程序基础

自己一直逃避的面向对象他又来了

最后跑去看了黑马的教程

https://github.com/yhzgithub/Personal-Bookshelves/blob/master/Java%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%20%E5%8D%B71%20%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86%20%E5%8E%9F%E4%B9%A6%E7%AC%AC9%E7%89%88%20%E5%AE%8C%E6%95%B4%E4%B8%AD%E6%96%87%E7%89%88%20.pdf

然后又学了这本书

https://github.com/deyou123/corejava/blob/master/Prentice.Hall.Core.Java.Volume.I.Fundamentals.11th.Edition.pdf

## Integer

```java
void should_take_care_of_number_type_when_doing_calculation() {
    final double result1 = 2 / 3 * 5;
    final double result2 = 2 * 5 / 3;

    // TODO:
    //  please modify the following lines to pass the test.
    //
    // Hint:
    //  If you want some reference please see page 59 of "Core Java Vol 1", section 3.5.2.
    // <!--start
    final double expectedResult1 = 0;
    final double expectedResult2 = 3;
    // --end-->

    assertEquals(expectedResult1, result1, +1.0E-05);
    assertEquals(expectedResult2, result2, +1.0E-05);
```

```java
void should_truncate_number_when_casting() {
    final int integer = 0x0123_4567;
    final short smallerInteger = (short)integer;

    // TODO:
    //  please modify the following lines to pass the test. Please refer to page 60 of "Core Java Vol 1", section 3.5.3.
    // <!--start
    final short expected = 0x4567;//TODO ?
    // --end-->

    assertEquals(expected, smallerInteger);
}
```

```java
void should_increment() {
    int integer = 3;

    int result = integer++;

    // TODO:
    //  please modify the following code to pass the test. You should write the
    //  result directly.
    // <--start
    final int expectedCurrentInteger = 4;
    final int expectedResult = 3;//TODO ?
    // --end-->

    assertEquals(expectedCurrentInteger, integer);
    assertEquals(expectedResult, result);
}
    void should_increment_2() {
        int integer = 3;

        int result = ++integer;

        // TODO:
        //   please modify the following code to pass the test. You should write the
        //   result directly.
        // <--start
        final int expectedCurrentInteger = 4;
        final int expectedResult = 4;
        // --end-->

        assertEquals(expectedCurrentInteger, integer);
        assertEquals(expectedResult, result);
    }
```

## String

```java
@SuppressWarnings({"unused"})
@Test
void should_break_string_into_words_customized() {
    final String sentence = "This/is/Mike";

    // TODO: Extract words in the sentence.
    // <--Start
    String[] words = sentence.split("/");//TODO \\/ or /?
    // --End-->

    assertArrayEquals(new String[] {"This", "is", "Mike"}, words);
}

```

## Some tricks

```java
//iterate over an array and turn each element into its string representation.
    StringBuilder stringBuilder = new StringBuilder();
        for (int i : originalArray) {
            stringBuilder.append(i);
        }
        destination = stringBuilder.toString();
```

```java
 //Please reverse the array to a new one. You should not modify original array.
             reversed = Arrays.copyOf(originalArray, originalArray.length);
        Collections.reverse(Arrays.asList(reversed));//TODO Collection api
```



# 正则表达式

https://www.liaoxuefeng.com/wiki/1252599548343744/1304066080636961

这个更全面一点

https://www.runoob.com/regexp/regexp-syntax.html

# 面向对象

https://houbb.github.io/2020/07/19/java-basic-01-what-is-oo

我搁现在还是有点看不懂面向对象

## Inheritance

```java
void should_be_derived_from_object_class() {
    // TODO:
    //  Please write down the class type directly.
    //
    // Hint:
    //  If you find it difficult, please check the page 228 of "Core Java Vol 1", section 5.2.
    // <--start
    final Class<?> expectedSuperClass = Object.class;
    // --end-->

    assertEquals(expectedSuperClass, SimpleEmptyClass.class.getSuperclass());
}
```

```java
void should_call_super_class_constructor() {
    DerivedFromSuperClassWithDefaultConstructor instance = new DerivedFromSuperClassWithDefaultConstructor();

    // TODO:
    //  You should write the answer directly.
    //
    // Hint:
    //  If you find it difficult, please check page 207 of "Core Java Vol 1", section 5.1.3.
    // <--start
    final String[] expected = {"SuperClassWithDefaultConstructor.constructor()", "DerivedFromSuperClassWithDefaultConstructor.constructor()"};
    // --end-->

    String[] logs = instance.getLogger();

    assertArrayEquals(expected, logs);
}
```

```java
void should_call_super_class_constructor_continued() {
    DerivedFromSuperClassWithDefaultConstructor instance = new DerivedFromSuperClassWithDefaultConstructor(42);

    // TODO:
    //  You should write the answer directly.
    //
    // Hint:
    //  If you find it difficult, please check page 207 of "Core Java Vol 1", section 5.1.3.
    // <--start
    final String[] expected = {"SuperClassWithDefaultConstructor.constructor()",
            "DerivedFromSuperClassWithDefaultConstructor.constructor()",
            "DerivedFromSuperClassWithDefaultConstructor.constructor(int)"};
    // --end-->

    String[] logs = instance.getLogs();

    assertArrayEquals(expected, logs);
}
```

```java
void should_call_super_class_constructor_more() {
    DerivedFromSuperClassWithDefaultConstructor instance = new DerivedFromSuperClassWithDefaultConstructor("God");

    // TODO:
    //  You should write the answer directly.
    //
    // Hint:
    //  If you find it difficult, please check page 207 of "Core Java Vol 1", section 5.1.3.
    // <--start
    final String[] expected = {"SuperClassWithDefaultConstructor.constructor(String)",
            "DerivedFromSuperClassWithDefaultConstructor.constructor(String)"};
    // --end-->

    String[] logs = instance.getLogs();

    assertArrayEquals(expected, logs);
}
```

```java
void should_use_caution_when_dealing_with_array_type() {
    DerivedFromSuperClassWithDefaultConstructor[] array = new DerivedFromSuperClassWithDefaultConstructor[4];
    SuperClassWithDefaultConstructor[] arrayWithBaseType = (SuperClassWithDefaultConstructor[])array;

    boolean willThrow = false;

    try {
        arrayWithBaseType[arrayWithBaseType.length - 1] = new SuperClassWithDefaultConstructor();
    } catch (Exception error) {
        willThrow = true;
    }

    // TODO:
    //  You should write the answer directly.
    //
    // Hint:
    //  If you meet difficulties, you can refer to page 213 of "Core Java Vol 1", section 5.1.5.
    // <--start
    final boolean expected = true;
    // --end-->

    assertEquals(expected, willThrow);
}
```

## Innerclass

```java
package com.tw.javabasic;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.tw.javabasic.util.AnonymousClassUpdateField;
import com.tw.javabasic.util.InnerClassUpdateField;
import com.tw.javabasic.util.LocalClassUpdateField;
import com.tw.javabasic.util.StaticInnerClass;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class InnerClassTest {
    // Recommended time used: 30 min

    @Test
    void should_access_instance_field_of_parent_class() {
        InnerClassUpdateField instance = new InnerClassUpdateField();
        instance.somethingHappen();

        // TODO:
        //  Please write down your answer directly.
        //
        // Hint:
        //  If you want some references, please check page 331 of "Core Java Vol 1", section 6.4.1.
        // <--start
        final Integer expected = 2019;
        // --end-->

        assertEquals(expected, instance.getYear());
    }

    @Test
    void should_refer_inner_class_from_outside() {
        InnerClassUpdateField instance = new InnerClassUpdateField();

        InnerClassUpdateField.YearIncrementer incrementer = instance.new YearIncrementer();
        incrementer.increment();

        // TODO:
        //  Please write down your answer directly.
        //
        // Hint:
        //  And if you want some references, please check page 334 of "Core Java Vol 1", section 6.4.2.
        // <--start
        final Integer expected = 2019;
        // --end-->

        assertEquals(expected, instance.getYear());
    }

    @Test
    void should_update_field_using_local_class() {
        LocalClassUpdateField instance = new LocalClassUpdateField();
        instance.somethingHappen();

        // TODO:
        //  Please write down your answer directly.
        //
        // Hint:
        //  And if you want some references, please check page 331 of "Core Java Vol 1", section 6.4.1.
        //  and page 339, section 6.4.4.
        // <--start
        final Integer expected = 2019;
        // --end-->

        assertEquals(expected, instance.getYear());
    }

    @Test
    void should_update_field_using_anonymous_class() {
        AnonymousClassUpdateField instance = new AnonymousClassUpdateField();
        instance.somethingHappen();

        // TODO:
        //  Please write down your answer directly.
        //
        // Hint:
        //  And if you want some references, please check page 342 of "Core Java Vol 1", section 6.4.6.
        // <--start
        final Integer expected = 2019;
        // --end-->

        assertEquals(expected, instance.getYear());
    }

    @Test
    void should_create_instance_for_static_inner_class() {
        StaticInnerClass instance = new StaticInnerClass();
        StaticInnerClass.Inner inner = instance.createInner();

        // TODO:
        //  Please write down your answer directly.
        //
        // Hint:
        //  And if you want some references, please check page 346 of "Core Java Vol 1", section 6.4.7.
        // <--start
        final String expected = "Hello";
        // --end-->

        assertEquals(expected, inner.getName());
    }
}
```

## Interface

```java
package com.tw.java;

import com.tw.java.util.InterfaceExtendsInterfaceWithDefaultMethod;
import com.tw.java.util.InterfaceExtendsInterfaceWithDefaultMethodImpl;
import com.tw.java.util.InterfaceWithDefaultMethodImpl;
import com.tw.java.util.InterfaceWithOverrideDefaultImpl;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class InterfaceTest {
    // Recommended time used: 20 min

    @Test
    void should_support_default_method() {
        InterfaceWithDefaultMethodImpl instance = new InterfaceWithDefaultMethodImpl();

        // TODO:
        //  Please write down your answer directly.
        //
        // Hint:
        //  If you find it difficult, please check page 298 of "Core Java Vol 1", section 6.1.5.
        // <--start
        final String expected = "The truth of the universe is 42";
        // --end-->

        assertEquals(expected, instance.tellMeTheTruthOfTheUniverse());
    }

    @Test
    void should_choose_override_method() {
        InterfaceWithOverrideDefaultImpl instance = new InterfaceWithOverrideDefaultImpl();

        // TODO:
        //  Please write down your answer directly.
        //
        // Hint:
        //  If you find it difficult, please check page 298 of "Core Java Vol 1", section 6.1.5 and 6.1.6
        // <--start
        final String expected = "The truth of the universe is Anime";
        // --end-->

        assertEquals(expected, instance.tellMeTheTruthOfTheUniverse());
    }

    @Test
    void should_choose_override_method_continued() {
        InterfaceExtendsInterfaceWithDefaultMethod instance = new InterfaceExtendsInterfaceWithDefaultMethodImpl();

        // TODO:
        //  Please write down your answer directly.
        //
        // Hint
        //  If you find it difficult, please check page 298 of "Core Java Vol 1", section 6.1.5 and 6.1.6
        // <--start
        final String expected = "The truth of the universe is Game";
        // --end-->

        assertEquals(expected, instance.tellMeTheTruthOfTheUniverse());
    }
}

```

## Object

```java
void should_get_initialization_ordering() {
    InitializationOrderClass.resetLogs();
    InitializationOrderClass instance = new InitializationOrderClass();
    String[] logs = InitializationOrderClass.getLogs();

    // TODO:
    //  Please write down your answer directly.
    //
    // Hint
    //  If you find it difficult, please check page 172 of "Core Java Vol 1", section 4.6
    // <--start
    final String[] expected = {
            "Field Initializer",
            "Initialization Block",
            "Constructor with argument",
            "Default constructor"};
    // --end-->

    assertArrayEquals(expected, logs);
}
```



## 

# Collection

```java
//turn array collection into string arrays.
//https://www.baeldung.com/java-collection-toarray-methods.
array=arrayCollection.toArray(new String[0]);
```

```java
void should_remove_duplication_from_a_list() {
    List<String> listWithDuplication = Arrays.asList(
        "tiger", "monkey", "tiger", "panda", "monkey");
    List<String> withoutDuplication = null;

    // TODO: Remove duplications in `listWithDuplication` and please reserve the
    //   original order. You should not use Streaming API.
    // <-start-
    withoutDuplication = new ArrayList<>();
    for (String item : listWithDuplication) {
        if (!withoutDuplication.contains(item)) {
            withoutDuplication.add(item);
        }
    }

    // withoutDuplication = new ArrayList<>(new LinkedHashSet<>(listWithDuplication));
    // --end->

    assertIterableEquals(Arrays.asList("tiger", "monkey", "panda"), withoutDuplication);
}
```

```java
void should_iterate_over_an_iterable() {
    final Iterator<String> iterator = Arrays.asList("tiger", "monkey", "panda").iterator();
    final List<String> cloned = new ArrayList<>();

    // TODO: Please iterate over the `iterable` and turn them into upper-case
    //   words. You should not use Streaming API.
    //
    // Hint:
    //  If you meet difficulties, please refer to "Core Java Vol 1", section 9.1.3.
    // <-start-
    while (iterator.hasNext()) {
        cloned.add(iterator.next().toUpperCase());
    }
    // --end-->

    assertIterableEquals(
        Arrays.asList("TIGER", "MONKEY", "PANDA"),
        cloned);
}
```

```java
void should_create_sorted_collection() {
    final List<Integer> unsorted = Arrays.asList(1, 6, 2, 4, 33, 76, 8, 9);
    List<Integer> sorted = null;

    // TODO: Please create new sorted list. You should use existed method to
    //   do sorting work. You should not use Streaming API.
    // <-start-
    sorted = new ArrayList<>(unsorted);
    Collections.sort(sorted);
    // --end-->

    assertEquals(Arrays.asList(1, 6, 2, 4, 33, 76, 8, 9), unsorted);
    assertEquals(Arrays.asList(1, 2, 4, 6, 8, 9, 33, 76), sorted);
}
```



# 其他

牛哇

- 遇见...就停止，可以不用if，而是用for+break例如遇见13后面的数就不处理，可以先不检测13在哪，而是先处理，直到碰到13

//TODO :

- IllegalArgumentException

# Adavanced

https://github.com/deyou123/corejava/blob/master/Prentice.Hall.Core.Java.Volume.II.Advanced.Features.11th.Edition.pdf

## Exception Handling





# 反馈

驼峰命名法 https://baike.baidu.com/item/%E9%A9%BC%E5%B3%B0%E5%91%BD%E5%90%8D%E6%B3%95/7560610

工厂模式 https://www.runoob.com/design-pattern/factory-pattern.html

常用的变量使用ctral+alt+v(command+option+v)抽取出来

if else的问题

https://www.bilibili.com/video/BV1eP4y157Jp/?vd_source=d6b48e0986e130482e80a9999aa637b7

# 一些遗漏知识点

## implements vs extends

https://stackoverflow.com/questions/10839131/implements-vs-extends-when-to-use-whats-the-difference

`extends` is for *extending* a class.

`implements` is for *implementing* an interface

The difference between an interface and a regular class is that in an interface you can not implement any of the declared methods. Only the class that "implements" the interface can implement the methods. The C++ equivalent of an interface would be an abstract class (not EXACTLY the same but pretty much).

Also java doesn't support **multiple inheritance** for classes. This is solved by using multiple interfaces.

```java
 public interface ExampleInterface {
    public void doAction();
    public String doThis(int number);
 }

 public class sub implements ExampleInterface {
     public void doAction() {
       //specify what must happen
     }

     public String doThis(int number) {
       //specfiy what must happen
     }
 }
```

now extending a class

```java
 public class SuperClass {
    public int getNb() {
         //specify what must happen
        return 1;
     }

     public int getNb2() {
         //specify what must happen
        return 2;
     }
 }

 public class SubClass extends SuperClass {
      //you can override the implementation
      @Override
      public int getNb2() {
        return 3;
     }
 }
```

in this case

```java
  Subclass s = new SubClass();
  s.getNb(); //returns 1
  s.getNb2(); //returns 3

  SuperClass sup = new SuperClass();
  sup.getNb(); //returns 1
  sup.getNb2(); //returns 2
```

Also, note that an `@Override` tag is not required for implementing an interface, as there is nothing in the original interface methods *to be overridden*

I suggest you do some more research on **dynamic binding, polymorphism and in general inheritance in Object-oriented programming**

## Overide vs Overload

https://www.runoob.com/java/java-override-overload.html

重写(Override)

重写是子类对父类的允许访问的方法的实现过程进行重新编写, 返回值和形参都不能改变。**即外壳不变，核心重写！**

重写的好处在于子类可以根据需要，定义特定于自己的行为。 也就是说子类能够根据需要实现父类的方法。

重写方法不能抛出新的检查异常或者比被重写方法申明更加宽泛的异常。例如： 父类的一个方法申明了一个检查异常 IOException，但是在重写这个方法的时候不能抛出 Exception 异常，因为 Exception 是 IOException 的父类，抛出 IOException 异常或者 IOException 的子类异常。

重载(Overload)

重载(overloading) 是在一个类里面，方法名字相同，而参数不同。返回类型可以相同也可以不同。

每个重载的方法（或者构造函数）都必须有一个独一无二的参数类型列表。

最常用的地方就是构造器的重载。

## 构造方法 private public

## this super

https://www.cnblogs.com/hasse/p/5023392.html

## 泛型

# 前后端联调的跨域问题

因为请求是从1234发出，目标是8080，会被浏览器限制

什么是跨域问题？由于浏览器的同源策略所导致的，在当前域的网页中，无法访问其他域的资源。

域：协议+域名+端口

三者完全相同才属于同源

比如

http://localhost:1234

为什么？安全

因为浏览器会存储一些安全信息在cookie session localstorage里

比如我们在访问

https://baidu.com

的时候同时也在访问

http://google.com/account

如果百度页面的js代码(JS都是在你本地浏览器运行的)请求了一个google的网页，那么就可以拿到cookie或者session里面的信息，造成泄密。

具体可看 跨域资源共享https://developer.mozilla.org/zh-CN/docs/Web/HTTP/CORS以及https://jishuin.proginn.com/p/763bfbd2f14a

https://juejin.cn/post/6844904055148380173

## Options请求

