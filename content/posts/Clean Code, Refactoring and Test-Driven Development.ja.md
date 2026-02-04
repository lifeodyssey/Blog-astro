---
title: 'クリーンコード、リファクタリング、テスト駆動開発'
tags:
  - Software Engineering
  - Software Design and Architecture
categories: 学習ノート
abbrlink: 5ffd7c75
slug: clean-code-refactoring-tdd
date: 2024-06-15 22:34:45
lang: ja
mathjax:
copyright:
---

ThoughtWorksでの勤務中に学んだコーディングの基本原則と習慣をまとめたいと思います。これらはいくつかの「形而上」のガイドラインで、後でKotlinを使用してデザインパターンを書き、その「形而下」の部分を実践する予定です。

GPTがほとんどのコードを書く手助けをする状況では、これらの原則はさらに重要になります。

<!-- more -->

# クリーンコード、リファクタリング、テスト駆動開発

参考資料：
- **『リファクタリング：既存のコードを安全に改善する』**
- **『Clean Code：アジャイルソフトウェア達人の技』**

## クリーンコード

**クリーンコード**が解決しようとする問題は：

> どんな馬鹿でも機械が理解できるコードを書ける。GPTはこの分野であなたより優れている。優れたプログラマーは人間が理解できるコードを書く。

> コードは主に人が読むために書かれ、たまたま機械が実行できるだけである。

> コードを読む時間は、書く時間の10倍以上かかる。

> 良い設計は一目瞭然で問題がなく、悪い設計には明らかな問題がない。

したがって、私たちが重視すべきは：
- **可読性！**
- **保守性！**

コードは**問題解決者**であるべきで、**トラブルメーカー**であってはならない。

**クリーンコード**は、上記の機能と価値を実現するコードであるべきです。

私のクリーンコードの基本ルールは：

- **1-10-50ルール**（まれな例外を除く）
    - 各メソッドは1レベル以上のインデントを持つべきではない。
        - try-catchとJavaScriptコールバックは例外。
    - 各メソッドは10行を超えるべきではない。
        - 中括弧とメソッド名自体は除く。
        - try-catchとAPI呼び出しは例外。
        - 複数行を無理に1行にまとめない。
    - 各クラスは50行を超えるべきではない。
        - import文はカウントしない。
- 適切な命名：変数、定数、メソッド、クラス、列挙値、ファイルなど。
- フォーマット
    - 変数
        - 変数宣言は使用箇所にできるだけ近くに配置
        - ローカル変数は関数の先頭に配置
        - ループ内の変数宣言は常にループ内で行う
        - インスタンス変数はクラスの先頭で宣言
    - メソッドの順序：ある関数が別の関数を呼び出す場合、一緒に配置し、呼び出し元を呼び出し先の上に配置
- コメントは「なし」。
- Elseは「なし」。
    - 早期リターンパターンを優先。

## リファクタリング

リファクタリングはクリーンコードを実現する方法です。

### コードの臭い

参考：[コードの臭い](https://refactoring.guru/refactoring/smells)

1. 重複コード
2. 長すぎるメソッド
3. 大きすぎるクラス
4. 長すぎるパラメータリスト
5. 基本データ型への執着
6. データの群れ
7. Switch文
8. 機能の横恋慕
9. コメント

5-9のKotlin例：

```kotlin
// 基本データ型への執着
val price: Double = 19.99
val currency: String = "USD"  // 通貨型の代わりに文字列を使用

// より良いアプローチ
data class Price(val amount: Double, val currency: Currency)
enum class Currency {
    USD, EUR, JPY
}
```

```kotlin
// データの群れ
fun processUser(firstName: String, lastName: String, address: String, city: String, zipCode: String) {
    // 処理コード
}

// より良いアプローチ
data class User(val firstName: String, val lastName: String, val address: Address)
data class Address(val streetAddress: String, val city: String, val zipCode: String)
```

```kotlin
// Switch文
fun calculateTax(productType: String, price: Double): Double {
    return when (productType) {
        "book" -> price * 0.05
        "food" -> price * 0.08
        "electronics" -> price * 0.15
        else -> price * 0.20
    }
}
```

```kotlin
// ポリモーフィズムを使用したより良いアプローチ
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
// 機能の横恋慕
class Order {
    fun totalPrice() = 20.0
}

class Payment {
    fun processPayment(order: Order) {
        val price = order.totalPrice()
        // 注文データに基づいて支払いを処理
    }
}
```

```kotlin
// より良いアプローチ：processPaymentメソッドをOrderクラスに移動
class Order {
    fun totalPrice() = 20.0

    fun processPayment() {
        val price = totalPrice()
        // 支払いを処理
    }
}
```

```kotlin
// コメント付きの悪い例
fun calculate() {
    // ユーザーがログインしていてセッションが有効かチェック
    if (user.isLoggedIn && session.isValid) {
        // 有効な場合、ユーザータイプに基づいて計算を実行
        // ユーザータイプが管理者の場合、10%増加
        if (user.type == "admin") {
            performCalculation() * 1.10
        } else {
            performCalculation()
        }
    }
}
```

```kotlin
// より良いアプローチ：コメント不要の自己説明的コード
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

### リファクタリング技法

参考：[リファクタリング技法](https://refactoring.guru/refactoring/techniques)

1. 変数の抽出
2. 一時変数のインライン化
3. メソッドの抽出
4. メソッドのインライン化

### いつリファクタリングが必要か？

- **コードレビュー**：コードレビュー中にコードの臭いを検出し、丁寧に改善を提案。
- **毎回のコミット**：各コミットで、コードを以前よりきれいにする。
- **読みにくいプロジェクトを引き継ぐとき**：リファクタリングを必要なタスクとして扱うようプロジェクトチームを説得。
- **イテレーション効率が期待を下回るとき**：リファクタリングを特定のタスクとして扱い、必要に応じて要件のイテレーションを一時停止。

### リファクタリングのルール

- キーボードショートカットを使用。
- **リファクタリングはコードの機能を壊してはならない；常にコンパイルと実行ができる状態を維持。**
- **リファクタリング中に新しいコードを書き始めることを避ける。**

## テスト駆動開発

**リファクタリングがコードの機能を壊さない**ことを確認するために、その完全性を検証する信頼できる方法が必要です。ここでテスト駆動開発（TDD）が価値を発揮します。TDDを統合することで、リファクタリングプロセス全体を通じてコードが期待通りに動作し続けることを確認するテストのセーフティネットを確立します。

![TDDサイクル](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/TDD%20cycle.png)

### TDDの3つの法則

1. 失敗したテストを通すため以外に、プロダクションコードを書いてはならない
2. 失敗するのに十分な以上のユニットテストを書いてはならない；コンパイル失敗も失敗である
3. 現在失敗しているユニットテストを通すのに十分な以上のプロダクションコードを書いてはならない

### テストダブルとテスト構造

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
        // Arrange (Given) - 準備
        val mockProcessor = mockk<CreditCardProcessor>()
        val mockTransactionLog = mockk<TransactionLog>(relaxed = true)

        every { mockProcessor.chargeCard(any(), any()) } returns true

        val paymentService = PaymentService(mockProcessor, mockTransactionLog)

        // Act (When) - 実行
        val paymentResult = paymentService.processPayment("1234567890", 100.0)

        // Assert (Then) - 検証
        assert(paymentResult)
        verify { mockTransactionLog.logTransaction("Success") }
    }
}
```

- **スタブ（Stub）**：上記のテストでは、`mockProcessor`がスタブとして使用されています。`chargeCard`メソッドが呼び出されたときに特定の応答（この場合は`true`）を返すように設定されています。スタブはテスト中のメソッド呼び出しに対して事前に決められた応答を提供するために使用されます。
- **モック（Mock）**：`mockTransactionLog`がモックとして使用されています。`relaxed = true`により事前定義された動作を提供するためスタブとも見なせますが、重要なのは事後に動作を検証していることです。`logTransaction`メソッドが正しい引数（"Success"）で呼び出されたかどうかを確認しています。これは典型的なモックの動作で、特定のメソッドが正しく呼び出されたかを検証することに重点を置いています。

より詳細な説明（参考：[Mocks Aren't Stubs](https://www.martinfowler.com/articles/mocksArentStubs.html)）：

- **ダミー（Dummy）**：渡されるが実際には使用されない。通常、パラメータリストを埋めるためだけに使用。
- **フェイク（Fake）**：実際に動作する実装を持つが、通常は本番環境に適さないショートカットを使用（[インメモリデータベース](https://www.martinfowler.com/bliki/InMemoryTestDatabase.html)が良い例）。
- **スタブ（Stub）**：テスト中の呼び出しに対して事前に用意された回答を提供し、通常はテスト用にプログラムされた内容以外には応答しない。
- **スパイ（Spy）**：スタブでもあるが、呼び出し方法に基づいて情報を記録する。送信されたメッセージ数を記録するメールサービスがその一例。
- **モック（Mock）**：ここで議論しているもの：期待値が事前にプログラムされたオブジェクトで、受け取ることが期待される呼び出しの仕様を形成する。

### テストピラミッド

![Webアプリケーションの例](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/20240615215559.png)

![テストピラミッド](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/20240615215639.png)
