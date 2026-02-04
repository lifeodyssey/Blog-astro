---
title: 不思議なKotlin+SpringBoot起動バグ
tags: 'Software Engineering'
categories: 学習ノート
abbrlink: 8bf3d056
mathjax:
date: 2022-12-03 21:37:35
copyright:
lang: ja
---
このバグに長い間悩まされました。
<!-- more -->
使用したbuild.gradle設定：
```kotlin
plugins {
    id("org.springframework.boot") version "2.7.5"
    kotlin("jvm") version "1.6.21"
    // ...
}
```

Dockerfileの内容：
```Dockerfile
FROM openjdk:11
COPY  /build/libs/demo-0.0.1-SNAPSHOT.jar Demo-0.0.1.jar
EXPOSE 8000
ENTRYPOINT ["java","-jar","/Demo-0.0.1.jar"]
```

Docker起動コマンド：
```bash
docker build -t demo .
docker run -dp 8000:8000 demo:latest
```

バグの症状：
1. ./gradlew bootRunでプロジェクトは正常に起動しアクセス可能
2. docker logs containerIDで正常な起動ログが見えるが、アクセス不可
3. java -jarでも正常に起動しアクセス可能

Dockerfileに問題がないことを繰り返し確認した後、Docker自体の問題だと判断しました。

これで思い出したのは、実は私のPCには2つのDockerがインストールされていたこと - colimaとdocker desktopです。

docker desktopを削除してbrewで再インストールしたら解決しました。
