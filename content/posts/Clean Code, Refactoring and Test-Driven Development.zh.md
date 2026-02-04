---
title: '整洁代码、重构与测试驱动开发'
tags:
  - Software Engineering
  - Software Design and Architecture
categories: 学习笔记
abbrlink: 5ffd7c75
slug: clean-code-refactoring-tdd
date: 2024-06-15 22:34:45
lang: zh
mathjax:
copyright:
---

我想总结一下在ThoughtWorks工作期间学到的一些编码基本原则和习惯。这些是一些"形而上"的指导原则，之后我计划用Kotlin编写一些设计模式来实践"形而下"的部分。

在GPT可以帮助我们编写大部分代码的背景下，这些原则变得更加重要。

<!-- more -->

# 整洁代码、重构与测试驱动开发

参考资料：
- **《重构：改善既有代码的设计》**
- **《代码整洁之道》**

## 整洁代码

**整洁代码**要解决的问题是：

> 任何傻瓜都能写出机器能理解的代码。GPT在这方面比你更擅长。优秀的程序员写的是人类能理解的代码。

> 代码首先是写给人看的，只是顺便能让机器执行。

> 阅读代码所花的时间远远超过编写代码的时间，比例约为10:1。

> 好的设计一目了然没有问题，而糟糕的设计则没有明显的问题。

所以我们应该重视的是：
- **可读性！**
- **可维护性！**

代码应该是**问题解决者**，而不是**麻烦制造者**。

**整洁代码**应该是能够实现上述功能和价值的代码。

我的整洁代码基本规则是：

- **1-10-50规则**（极少数情况下允许例外）
    - 每个方法不应超过一层缩进。
        - try-catch和JavaScript回调除外。
    - 每个方法不应超过10行。
        - 不包括大括号和方法名本身。
        - try-catch和API调用除外。
        - 不要强行将多行合并为一行。
    - 每个类不应超过50行。
        - import语句不计入。
- 合理命名：变量、常量、方法、类、枚举值、文件等。
- 格式化
    - 变量
        - 变量声明应尽可能靠近其使用位置
        - 局部变量应出现在函数顶部
        - 循环内的变量声明应始终在循环内部
        - 实例变量应在类的顶部声明
    - 方法顺序：如果一个函数调用另一个函数，应放在一起，调用者应放在被调用者上方
- "不要"注释。
- "不要"Else。
    - 优先使用提前返回模式。

## 重构

重构是实现整洁代码的方式。

### 代码异味

参考：[代码异味](https://refactoring.guru/refactoring/smells)

1. 重复代码
2. 过长方法
3. 过大类
4. 过长参数列表
5. 基本类型偏执
6. 数据泥团
7. Switch语句
8. 依恋情结
9. 注释

5-9的Kotlin示例：

```kotlin
// 基本类型偏执
val price: Double = 19.99
val currency: String = "USD"  // 使用字符串而不是货币类型

// 更好的方式
data class Price(val amount: Double, val currency: Currency)
enum class Currency {
    USD, EUR, JPY
}
```

```kotlin
// 数据泥团
fun processUser(firstName: String, lastName: String, address: String, city: String, zipCode: String) {
    // 处理代码
}

// 更好的方式
data class User(val firstName: String, val lastName: String, val address: Address)
data class Address(val streetAddress: String, val city: String, val zipCode: String)
```

```kotlin
// Switch语句
fun calculateTax(productType: String, price: Double): Double {
    return when (productType) {
        "book" -> price * 0.05
        "food" -> price * 0.08
        "electronics" -> price * 0.15
        else -> price * 0.20
    }
}

// 使用多态的更好方式
interface Product {
    fun calculateTax(): Double
}

class Book(private val price: Double) : Product {
    override fun calculateTax() = price * 0.05
}

class Food(private val price: Double) : Product {
    override fun calculateTax() = price * 0.08
}

class Electronics(private val price: Double) : Product {
    override fun calculateTax() = price * 0.15
}
```

```kotlin
// 依恋情结
class Order {
    fun totalPrice() = 20.0
}

class Payment {
    fun processPayment(order: Order) {
        val price = order.totalPrice()
        // 基于订单数据处理支付
    }
}

// 更好的方式：将processPayment方法移到Order类中
class Order {
    fun totalPrice() = 20.0

    fun processPayment() {
        val price = totalPrice()
        // 处理支付
    }
}
```

```kotlin
// 带注释的不良实践
fun calculate() {
    // 检查用户是否已登录且会话有效
    if (user.isLoggedIn && session.isValid) {
        // 如果有效，根据用户类型执行计算
        // 如果用户类型是管理员，增加10%
        if (user.type == "admin") {
            performCalculation() * 1.10
        } else {
            performCalculation()
        }
    }
}

// 更好的方式：无需注释的自解释代码
fun calculate() {
    if (isValidSession()) {
        performUserSpecificCalculation()
    }
}

private fun isValidSession() = user.isLoggedIn && session.isValid
private fun performUserSpecificCalculation() = when (user.type) {
    "admin" -> performCalculation() * 1.10
    else -> performCalculation()
}
```

### 重构技术

参考：[重构技术](https://refactoring.guru/refactoring/techniques)

1. 提取变量
2. 内联临时变量
3. 提取方法
4. 内联方法

### 何时需要重构？

- **代码审查**：在代码审查中发现代码异味，礼貌地提出改进建议。
- **每次提交**：每次提交都应该让代码比之前更整洁。
- **接手难以阅读的项目时**：说服项目团队将重构作为必要任务。
- **迭代效率低于预期时**：将重构作为特定任务，必要时暂停需求迭代。

### 重构规则

- 使用键盘快捷键。
- **重构不应破坏代码功能；代码应始终能够编译和运行。**
- **避免在重构过程中开始编写新代码。**

## 测试驱动开发

为了确保**重构不会破坏代码功能**，我们需要一种可靠的方法来验证其完整性。这就是测试驱动开发（TDD）的价值所在。通过集成TDD，我们建立了一个测试安全网，确认代码在整个重构过程中继续按预期执行。

![TDD循环](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/TDD%20cycle.png)

### TDD三定律

1. 除非是为了让失败的测试通过，否则不允许编写任何生产代码
2. 不允许编写超过足以导致失败的单元测试；编译失败也算失败
3. 不允许编写超过足以通过当前失败单元测试的生产代码

### 测试替身与测试结构

```kotlin
interface CreditCardProcessor {
    fun chargeCard(cardInfo: String, amount: Double): Boolean
}

interface TransactionLog {
    fun logTransaction(status: String)
}

class PaymentService(
    private val processor: CreditCardProcessor,
    private val transactionLog: TransactionLog
) {
    fun processPayment(cardInfo: String, amount: Double): Boolean {
        val success = processor.chargeCard(cardInfo, amount)
        if (success) {
            transactionLog.logTransaction("Success")
        } else {
            transactionLog.logTransaction("Failure")
        }
        return success
    }
}
```

```kotlin
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.Test

class PaymentServiceTest {

    @Test
    fun `Given credit card is valid, When processing payment, Then log success`() {
        // Arrange (Given) - 准备
        val mockProcessor = mockk<CreditCardProcessor>()
        val mockTransactionLog = mockk<TransactionLog>(relaxed = true)

        every { mockProcessor.chargeCard(any(), any()) } returns true

        val paymentService = PaymentService(mockProcessor, mockTransactionLog)

        // Act (When) - 执行
        val paymentResult = paymentService.processPayment("1234567890", 100.0)

        // Assert (Then) - 断言
        assert(paymentResult)
        verify { mockTransactionLog.logTransaction("Success") }
    }
}
```

- **桩（Stub）**：在上面的测试中，`mockProcessor`被用作桩。它被配置为在调用`chargeCard`方法时返回特定响应（本例中为`true`）。桩用于在测试期间为方法调用提供预定响应。
- **模拟（Mock）**：`mockTransactionLog`被用作模拟。虽然它也可以被视为桩（因为`relaxed = true`提供了预定义行为），但关键是我们在事后验证其行为。我们检查`logTransaction`方法是否使用正确的参数（"Success"）被调用。这是典型的模拟行为，重点是验证某些方法是否被正确调用。

更详细的说明（参考：[Mocks Aren't Stubs](https://www.martinfowler.com/articles/mocksArentStubs.html)）：

- **哑对象（Dummy）**：被传递但从未实际使用。通常只是用来填充参数列表。
- **伪对象（Fake）**：有实际工作的实现，但通常采用某些捷径使其不适合生产环境（[内存数据库](https://www.martinfowler.com/bliki/InMemoryTestDatabase.html)是一个很好的例子）。
- **桩（Stub）**：为测试期间的调用提供预设答案，通常不响应测试中未编程的任何内容。
- **间谍（Spy）**：也是桩，但会根据调用方式记录一些信息。一种形式可能是记录发送了多少消息的电子邮件服务。
- **模拟（Mock）**：这里讨论的对象：预先编程了期望的对象，这些期望构成了它们预期接收的调用规范。

### 测试金字塔

![Web应用示例](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/20240615215559.png)

![测试金字塔](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/20240615215639.png)
