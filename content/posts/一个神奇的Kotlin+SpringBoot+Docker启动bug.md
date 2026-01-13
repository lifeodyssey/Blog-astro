---
title: 一个神奇的Kotlin+SpringBoot启动bug
tags: 'Software Engineering'
categories: 学习笔记
abbrlink: 8bf3d056
mathjax:
date: 2022-12-03 21:37:35
copyright:
---
这个bug卡了我好久了
<!-- more -->
所使用的build.gradle配置如下
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
    implementation("org.springframework.boot:spring-boot-starter-data-mongodb")  
    implementation("org.jetbrains.kotlin:kotlin-reflect")  
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")  
    compileOnly("org.projectlombok:lombok")  
    developmentOnly("org.springframework.boot:spring-boot-devtools")  
    annotationProcessor("org.springframework.boot:spring-boot-configuration-processor")  
    annotationProcessor("org.projectlombok:lombok")  
    testImplementation("org.springframework.boot:spring-boot-starter-test")  
    implementation("org.springframework.boot:spring-boot-starter-web")  
    implementation("org.springframework.boot:spring-boot-starter")  
    implementation("org.springframework.boot:spring-boot-starter-mustache")  
    implementation("org.jetbrains.kotlin:kotlin-reflect")  
  
    testImplementation(kotlin("test"))  
}  
  
tasks.withType<KotlinCompile> {  
    kotlinOptions {  
        freeCompilerArgs = listOf("-Xjsr305=strict")  
        jvmTarget = "11"  
    }  
}  
  
tasks.withType<Test> {  
    useJUnitPlatform()  
}  
tasks.test {  
    finalizedBy(tasks.jacocoTestReport) // report is always generated after tests run  
}  
  
tasks.jacocoTestReport {  
    dependsOn(tasks.test)  
    reports {  
        xml.required.set(false)  
        csv.required.set(false)  
        html.outputLocation.set(layout.buildDirectory.dir("jacocoHtml"))  
    }  
}  
tasks.jacocoTestCoverageVerification {  
    violationRules {  
        rule {  
            limit {  
                minimum = "0.8".toBigDecimal()  
            }  
        }    }}  
  
application {  
    mainClass.set("com.example.demo.DemoApplicationKt")  
}  
tasks.build {  
    dependsOn(tasks.jacocoTestCoverageVerification)  
}  
jacoco {  
    applyTo(tasks.run.get())  
}  
tasks.register<JacocoReport>("applicationCodeCoverageReport") {  
    executionData(tasks.run.get())  
    sourceSets(sourceSets.main.get())  
}
```
Dockefile内容为下 
```Dockerfile
FROM openjdk:11  
COPY  /build/libs/demo-0.0.1-SNAPSHOT.jar Demo-0.0.1.jar  
EXPOSE 8000  
ENTRYPOINT ["java","-jar","/Demo-0.0.1.jar"]
```
docker启动命令如下
```bash
docker build -t demo .
docker run -dp 8000:8000 demo:latest
```
bug表现为
1. 使用./gradlew bootRun可以正常启动项目并且访问
2. 使用docker logs containerID能看到正常启动的log，但是无法访问
3. 使用java -jar也可以正常启动项目并访问 

在反复确认Dockerfile没有问题之后，确定为docker本身的问题。
这不由得让我想起来，我的电脑上其实是装了两个Docker的，一个是colima，一个是docker desktop。
所以我删除docker desktop之后用brew重新装了一个就好了