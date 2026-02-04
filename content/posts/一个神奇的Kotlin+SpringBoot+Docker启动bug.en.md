---
title: A Mysterious Kotlin+SpringBoot Startup Bug
tags: 'Software Engineering'
categories: Study Notes
abbrlink: 8bf3d056
mathjax:
date: 2022-12-03 21:37:35
copyright:
lang: en
---
This bug stuck me for a long time.
<!-- more -->
The build.gradle configuration used is as follows:
```kotlin
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("org.springframework.boot") version "2.7.5"
    id("io.spring.dependency-management") version "1.0.15.RELEASE"
    kotlin("jvm") version "1.6.21"
    kotlin("plugin.spring") version "1.6.21"
    kotlin("plugin.jpa") version "1.6.21"
    id("org.jlleitschuh.gradle.ktlint") version "11.0.0"
    jacoco
    application
}

group = "com.example"
version = "0.0.1-SNAPSHOT"
java.sourceCompatibility = JavaVersion.VERSION_11

configurations {
    compileOnly {
        extendsFrom(configurations.annotationProcessor.get())
    }
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-data-jdbc")
    implementation("org.springframework.boot:spring-boot-starter-data-jpa")
    // ... more dependencies
}
```

Dockerfile content:
```Dockerfile
FROM openjdk:11
COPY  /build/libs/demo-0.0.1-SNAPSHOT.jar Demo-0.0.1.jar
EXPOSE 8000
ENTRYPOINT ["java","-jar","/Demo-0.0.1.jar"]
```

Docker startup commands:
```bash
docker build -t demo .
docker run -dp 8000:8000 demo:latest
```

Bug symptoms:
1. Using ./gradlew bootRun starts the project normally and it's accessible
2. Using docker logs containerID shows normal startup logs, but the app is inaccessible
3. Using java -jar also starts the project normally and it's accessible

After repeatedly confirming the Dockerfile had no issues, I determined it was a Docker problem itself.

This reminded me that I actually had two Dockers installed on my computer - one was colima, and one was docker desktop.

So after deleting docker desktop and reinstalling with brew, it worked.
