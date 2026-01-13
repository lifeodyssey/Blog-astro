---
title: Obsidian Daily Record System
tags:
  - 随便写写
categories: 实用工具
abbrlink: f36cc8b7
date: 2022-07-23 23:36:22
mathjax:
copyright:
---

这一篇的后续https://lifeodyssey.github.io/posts/b58cbef5.html
先写一下手上这套方案的缺陷
- 只能做日常记录，没有做到Personal Knowledge Management (PKM)
- 因为没有购买obsidian的付费服务，依赖于git进行同步，无法做到全平台同步，只能在桌面端使用，移动端还是依赖于便签app
- 对任务清单的支持不好，这部分还是用Microsoft ToDo解决的

日后或许会把他和nextcloud结合起来
主要根据这个完成的https://diygod.me/obsidian/

<!-- more -->

# 需要解决的问题

相比于原作者，我不喜欢把很多东西都加到Obsidian里，像运动健身我在google fit，睡眠在android睡眠伴侣，钱在薄荷记账。

我这里用Obsidian主要完成下面几件事

1. 每天的Completed List，写出来十件事来夸夸自己
2. 一些日常生活中可能会想到的小日记，大型日记写在了onenote和纸质日记，就相当于一个记事本
3. Kanban，因为不想用trello

所以对原来的版本做了很多简化。

需求主要有如下几个

1. 每天点击Calendar自动生成日记并且按照/Year/Month/Week的顺序排列
2. 每个星期生成带甘特图的计划表和toggle总结
3. 每个月自动生成kanban
4. 每年生成一个大的带甘特图的计划表，并且有一个提醒我在年底回顾的艾特

一个个来

# 具体改动

这个利用Periodic Note和templater里的function就可以完成，因为我动不动就开个加速器，就没有采用获取IP的方式来获取当前位置，而是用了固定地点，以getWeather为例

```bash
curl wttr.in/"$(curl -s --header "user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36" https://api.ip.sb/geoip | /opt/homebrew/bin/jq -r ".city" | sed 's/ /%20/')"\?format="%l+%c%t"
```

我把他改成了

```bash

curl wttr.in/City:Province:Country?format="%l+%c%t"
```

其他的没变
