---
title: SQL
tags:
  - Java
  - 'Software Engineering'
categories: Study Notes
abbrlink: dbd66adf
password: woshiyelanxiaojiedeggou
date: 2022-06-04 18:34:45
mathjax:
copyright:
lang: en
---

Something I've been putting off is finally here.

<!-- more -->

# SQL Query

Query data: https://www.liaoxuefeng.com/wiki/1177760294764384/1179610544539040

SQL Must Know chapters 9-14

When querying from multiple tables, always specify which tables.

Example:

```sql
SELECT DISTINCT `customers`.`country`, `customers`.`city`
FROM `customers`
WHERE `customers`.`country` = 'France' OR `customers`.`country` = 'UK'
ORDER BY `customers`.`country`, `customers`.`city`;
```

MySQL doesn't support TOP, use LIMIT instead.

## Join

https://www.runoob.com/sql/sql-join.html

Inner Join: Both tables have matching keys
Left Join: Keep all from A, join B if exists, otherwise null

# JDBC

https://www.cainiaojc.com/jdbc/jdbc-tutorial.html

## ResultSet

The return value of statement.execute is ResultSet.

## Connection

I thought creating a Connection for each method was tedious, so I tried using one connection for all methods. This caused:

java.sql.SQLNonTransientConnectionException: No operations allowed after connection closed.

This is because try-with-resources automatically releases resources after the try block. Since I only created one connection, it was released after the first try-with-resources, and subsequent calls failed.
