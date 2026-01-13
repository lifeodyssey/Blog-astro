---
title: 'Clean Code, Refactoring and Test-Driven Development'
tags:
  - Software Engineering
  - Software Design and Architecture
categories: 学习笔记
abbrlink: 5ffd7c75
date: 2024-06-15 22:34:45
mathjax:
copyright:
---

I would like to summarize some of the basic principles and habits of coding that I learned while working at ThoughtWorks. These are some 'metaphysical' guidelines, and later I plan to use Kotlin to write some design patterns to practice the 'physical' aspects.

In a context where GPT can help us write most of the code, these principles become even more important. 


私はThoughtWorksでの勤務中に学んだコーディングの基本原則と習慣をまとめたいと思います。これらはいくつかの「形而上」のガイドラインで、後でKotlinを使用してデザインパターンを書き、その「形而下」の部分を実践する予定です。

GPTがほとんどのコードを書く手助けをする状況では、これらの原則はさらに重要になります。
<!-- more -->

# Clean Code, Refactoring and Test-Driven Development
Reference: 
- **Refactoring: Improving the Design of Existing Code**
- **Clean Code: A Handbook of Agile Software Craftsmanship**
  
  ## Clean Code
  
  The problem that **Clean Code** want to solve is raised
- 

  > Any fool can write code that a machine can understand. GPT is better at this field that you. Good programmers write code that humans can understand.
  
  
> Code is primarily written for people to read, and only incidentally for machines to execute.


> The time spent reading code far exceeds the time spent writing code by a factor of 10x.


> Good design is obvious with no issues, while bad design has no obvious issues.

So the things that we values should be 
- **Readability! **
- **Maintainability!**

Code should be a **Problem Solver**, not a *Trouble Maker** .

A **Clean Code** should be a code that achieve the beforehand mentioned functions and values.

My basic rules of Clean Code are 

- **1-10-50 Rule** (exceptions allowed in rare cases)
    - Each method should not have more than one level of indentation.
        - Exceptions for try-catch and JavaScript callbacks.
    - Each method should not exceed 10 lines.
        - Excluding braces and the name itself.
        - Exceptions for try-catch and fetching APIs.
        - Do not force multiple lines into a single line.
    - Each class should not exceed 50 lines.
        - Import statements do not count.
- Reasonable naming: variables, constants, methods, classes, enum values, files, etc.
- Formatting 
    - Variable
	    - Variable declarations should be as close as possible to their point of use
	    - Local variables should appear at the top of the function 
	    - Variable declarations within loops should always occur inside the loop
	    - Entity variables should be declared at the top of the class.
	- Method order: if one function calls another, should be placed together, and the caller should be placed above the callee
- "No" Comments.
- "No" Else.
    - Favor the Return Early pattern.


## Refactoring
Refactoring is the way to achieve clean code.
### Code Smell
Reference :[Code Smells](https://refactoring.guru/refactoring/smells) 

1. Duplicate Code
2. Long Method
3. Large Class
4. Long Parameter List
5. Primitive Obsession
6. Data Clumps
7. Switch Statements
8. Feature Envy
9. Comments

example for 5-9 using Kotlin 

```kotlin
// Primitive Obsession
val price: Double = 19.99
val currency: String = "USD"  // Using string instead of a currency type

// Better approach
data class Price(val amount: Double, val currency: Currency)
enum class Currency {
    USD, EUR, JPY
}

```

```kotlin
// Data Clumps
fun processUser(firstName: String, lastName: String, address: String, city: String, zipCode: String) {
    // Processing code
}

// Better approach
data class User(val firstName: String, val lastName: String, val address: Address)
data class Address(val streetAddress: String, val city: String, val zipCode: String)
```

```kotlin
// Switch Statements
fun calculateTax(productType: String, price: Double): Double {
    return when (productType) {
        "book" -> price * 0.05
        "food" -> price * 0.08
        "electronics" -> price * 0.15
        else -> price * 0.20
    }
}

// Better approach using polymorphism
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
// Feature Envy
class Order {
    fun totalPrice() = 20.0
}

class Payment {
    fun processPayment(order: Order) {
        val price = order.totalPrice()
        // Processing payment based on order's data
    }
}

// Better approach: moving the processPayment method to the Order class
class Order {
    fun totalPrice() = 20.0

    fun processPayment() {
        val price = totalPrice()
        // Processing payment
    }
}
```

```kotlin
// Bad practice with comments
fun calculate() {
    // Check if the user is logged in and the session is valid
    if (user.isLoggedIn && session.isValid) {
        // If valid, perform the calculation based on the user type
        // If the user type is admin, apply a 10% increase
        if (user.type == "admin") {
            performCalculation() * 1.10
        } else {
            performCalculation()
        }
    }
}

// Better approach: self-explanatory code without comments
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
### Refactoring Techniques
Reference : [Refactoring Techniques](https://refactoring.guru/refactoring/techniques)
1. Extract Variable
2. Inline Temp
3. Extract Method
4. Inline Method 

###  When is refactoring needed?
- **Code Review**: Detect bad smells during code reviews and politely suggest improvements.
- **Every Commit**: Each commit you make should leave the code cleaner than it was before.
- **When Taking Over a Difficult-to-Read Project**: Convince the project team to treat refactoring as a necessary task.
- **When Iteration Efficiency Is Below Expectations**: Treat refactoring as a specific task, and if necessary, pause to iterate on requirements.

### Rules of Refactoring
- Use keyboard shortcuts.
- **Refactoring should not break the functionality of the code; it should always be able to compile and run.**
- **Avoid starting to write new code while in the middle of refactoring.**
  
  ## Test-Driven-Development
  
 To ensure that **refactoring does not break the functionality of the code**, we require a reliable method to verify its integrity. This is where Test-Driven Development (TDD) proves invaluable. By integrating TDD, we establish a safety net of tests that confirm the code continues to perform as expected throughout the refactoring process.

![TDD cycle](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/TDD%20cycle.png)

### The 3 Laws of TDD 

1. You are not allowed  to write any production code unless it is to make a failed pass 
2. You are not allowed to write any more of a unit test than is sufficient to fail; and compilation failures are failures
3. You are not allowed to write any more production code than is sufficient to pass on failing unit test . 
   
### Test Double and Test Structure

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
        // Arrange (Given)
        val mockProcessor = mockk<CreditCardProcessor>()
        val mockTransactionLog = mockk<TransactionLog>(relaxed = true)
        
        every { mockProcessor.chargeCard(any(), any()) } returns true
        
        val paymentService = PaymentService(mockProcessor, mockTransactionLog)

        // Act (When)
        val paymentResult = paymentService.processPayment("1234567890", 100.0)

        // Assert (Then)
        assert(paymentResult)
        verify { mockTransactionLog.logTransaction("Success") }
    }
}


```

- **Stub**: In the test above, `mockProcessor` is used as a stub. It is configured to return a specific response (`true` in this case) when its `chargeCard` method is called. Stubs are used to provide predetermined responses to method calls during tests.
- **Mock**: `mockTransactionLog` is used as a mock. While it could also be seen as a stub because it is providing predefined behavior (due to `relaxed = true`), the key aspect here is that we're verifying its behavior post-factum. We are checking whether the `logTransaction` method was called with the correct argument ("Success"). This is typical mocking behavior, where the emphasis is on verifying that certain methods are called correctly.
  
  More detailed ( Reference:   [Mocks Aren't Stubs](https://www.martinfowler.com/articles/mocksArentStubs.html))
- **Dummy** objects are passed around but never actually used. Usually they are just used to fill parameter lists.
- **Fake** objects actually have working implementations, but usually take some shortcut which makes them not suitable for production (an [in memory database](https://www.martinfowler.com/bliki/InMemoryTestDatabase.html) is a good example).
- **Stubs** provide canned answers to calls made during the test, usually not responding at all to anything outside what's programmed in for the test.
- **Spies** are stubs that also record some information based on how they were called. One form of this might be an email service that records how many messages it was sent.
- **Mocks** are what we are talking about here: objects pre-programmed with expectations which form a specification of the calls they are expected to receive.
### Test Pyramid

![Example of a web application](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/20240615215559.png)

![Test Pyramid](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/20240615215639.png)


  
