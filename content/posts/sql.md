---
title: sql
tags:
  - Java
  - 'Software Engineering'
categories: 学习笔记
abbrlink: dbd66adf
password: woshiyelanxiaojiedeggou
date: 2022-06-04 18:34:45
mathjax:
copyright:
---

我一直懒得搞的东西她终于来了

<!-- more -->

# SQL Query



查询数据 https://www.liaoxuefeng.com/wiki/1177760294764384/1179610544539040

SQL必知必会第五版的9-14课

有从多个表里查询的东西一定写明是哪几个表

例如: 

```sql
/*
 * 请告诉我 'France' 和 'UK' 的顾客（customer）的国家（country）和城市（city）的组合方式（不重复）。
 * 查询结果应当包含如下信息：
 *
 * +──────────+───────+
 * | country  | city  |
 * +──────────+───────+
 *
 * 结果应当首先按照 country 排序，再按照 city 排序。
 */

SELECT DISTINCT `customers`.`country`, `customers`.`city`
FROM `customers`
WHERE `customers`.`country` = 'France' OR `customers`.`country` = 'UK'
ORDER BY `customers`.`country`, `customers`.`city`;
```



而不是直接讲一个country和city

只涉及到一个表的，不需要加，但是如果在多个表中拥有相同的关键词，可以加上是哪个表的来更清晰易读

mysql不支持top，支持limit

```sql
SELECT `orderNumber`,(`quantityOrdered`*`priceEach`) AS `subtotal`
FROM `orderdetails`
ORDER BY `subtotal` DESC
/*或者可以用order by `quantityOrdered`*`priceEach`
*/
LIMIT 5;
```

## Join



https://www.runoob.com/sql/sql-join.html



![img](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202206121504545.png)

Inner Join 要查的每一个都在两个表里面有 以相同的键值链接

Left join  A全部保留 B如果有的话就接上 没有就null

TODO: 有两个题还不太会

# JDBC

https://www.cainiaojc.com/jdbc/jdbc-tutorial.html 这个比廖雪峰讲的更清楚一点

https://www.runoob.com/mysql/mysql-create-tables.html　sql怎么建表

什么是 innodb https://wulijun.github.io/2012/09/29/mysql-innodb-intro.html

示例代码

```java
public TodoItemRepository(ServiceConfiguration configuration) {
    this.configuration = configuration;
}

/**
 * Create an item and save it into database. The `checked` status should default to `false`.
 *
 * @param name The name of the item.
 * @return The `id` of the item created.
 * @throws IllegalArgumentException The name is `null`.
 */
public long create(String name) throws Exception {
    // TODO:
    //   Please implement the method.
    // <-start-

    if (name == null) {
        throw new IllegalArgumentException("Name must be provided.");
    }

    try (
            Connection connection = createConnection();//看看能不能连接上数据库
            final PreparedStatement statement = connection.prepareStatement(
                    "INSERT INTO `todo_items` (`name`) VALUES (?)", Statement.RETURN_GENERATED_KEYS);//创建statement对象 https://juejin.cn/post/6844903584492945422 返回generated keys
    ) {
        statement.setString(1, name);
        statement.execute();//执行语句
        final ResultSet resultSet = statement.getGeneratedKeys();//生成一个result 可以认为是一行 resultset会指向statement的前一行
        resultSet.next();//前进到下一行
        return resultSet.getLong(1);//https://stackoverflow.com/questions/19873190/statement-getgeneratedkeys-method 得到属性为long的值
    }
}
    // --end-->
}

/**
 * Change item checked status to specified state.
 *
 * @param id The item to update.
 * @param checked The new checked status.
 * @return If the item exist returns true, otherwise returns false.
 */
public boolean changeCheckedStatus(long id, boolean checked) throws Exception {
    // TODO:
    //   Please implement the method.
    // <-start-
    try (
            Connection connection = createConnection();
            final PreparedStatement statement = connection.prepareStatement(
                    "UPDATE `todo_items` SET `checked`=? WHERE `id`=?");
    ) {
        statement.setBoolean(1, checked);
        statement.setLong(2, id);
        return statement.executeUpdate() > 0;//https://blog.csdn.net/u012501054/article/details/80323176 execute executUpdate区别
    }
    // --end-->
}

/**
 * Get item by its id.
 *
 * @param id The id of the item.
 * @return The item entity. If the item does not exist, returns empty.
 */
public Optional<TodoItem> findById(long id) throws Exception {
    // TODO:
    //   Please implement the method.
    // <-start-
    try (
            Connection connection = createConnection();
            final PreparedStatement statement = connection.prepareStatement(
                    "SELECT `name`, `checked` FROM `todo_items` WHERE `id`=?");
    ) {
        statement.setLong(1, id);
        final ResultSet resultSet = statement.executeQuery();
        if (resultSet.next()) {
            final String name = resultSet.getString("name");
            final boolean checked = resultSet.getBoolean("checked");
            return Optional.of(new TodoItem(id, name, checked));
        }

        return Optional.empty();
    }
}
    // --end-->
}
```

## resultset

statment.excute的返回值是resultset，可以看https://download.oracle.com/technetwork/java/javase/6/docs/zh/api/java/sql/ResultSet.html

这里想知道的是，怎么判断statement.excute是怎么成功的呢

![image-20220616110016249](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202206161100339.png)

![image-20220616110032433](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202206161100472.png)

看返回值

## Connection

我一直觉得每次运行小方法都要搞一个Connection conn = getConnection()非常的麻烦，所以尝试让所有的方法都用同一个connection，然后出现了这个错误

java.sql.SQLNonTransientConnectionException: No operations allowed after connection closed.

这是因为 try with resource 方法会在try的语句里执行结束之后自动释放所用到的资源，又因为我只创建了一个connection 所以在经过一个try with resource之后，我创建的connection就被释放了，后续的就都跑不出来。

我的这个想法还有一个问题就是 我创建的这个connection没有被释放 会引起资源泄露

