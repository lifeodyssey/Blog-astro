---
title: Windows环境下hexo博客搭建
tags:
  - 博客搭建
  - 入门
categories: 资料贴整理
abbrlink: b1f5683b
date: 2019-03-06 21:44:39
copyright: true
---
自己算来算去都搭建了三次博客了，前两次是在ubuntu上，第二次在ubuntu搭建的时候花费了好多时间解决node.js和npm的问题，索性这次就在windows上了，虽然windows的命令行用着很蛋疼，但是架不住方便啊。赶紧把博客搭出来写文章才是最主要的。

<!-- more -->

每次搭建都得花好多时间搜集资料贴，这次索性把资料贴整理出来，免得自己下次再去到处找。

## 博客生成

### 入门

Github Pages可以被认为是用户编写的、托管在github上的静态网页。使用Github Pages可以为你提供一个免费的服务器，免去了自己搭建服务器和写数据库的麻烦。此外还可以绑定自己的域名。因此，我们需要去[github官网](https://github.com/)注册一个账号。

Hexo 是一个快速、简洁且高效的博客框架。Hexo 使用 Markdown（或其他渲染引擎）解析文章，在几秒内，即可利用靓丽的主题生成静态网页。



### 安装环境

1.安装[git](https://git-scm.com/)

2.安装[node.js](https://nodejs.org/en/)

以上两步对于windows用户来说非常友好了，按照默认来装就可以了。

3.安装hexo

右键呼出git bash。输入：
``` bash
npm install -g hexo
npm install hexo-deployer-git --save 
```
然后输入`hexo -v` 出现一系列版本号就是安装成功了，像我这样
``` bash
$ hexo -v
hexo: 3.8.0
hexo-cli: 1.1.0
os: Windows_NT 10.0.17763 win32 x64
http_parser: 2.8.0
node: 10.15.3
v8: 6.8.275.32-node.51
uv: 1.23.2
zlib: 1.2.11
ares: 1.15.0
modules: 64
nghttp2: 1.34.0
napi: 3
openssl: 1.1.0j
icu: 62.1
unicode: 11.0
cldr: 33.1
tz: 2018e`
```
如果不成功的话可以同时按下win和R，输入cmd,分别使用如下三个命令，如果有一个没有返回版本信息则说明这个软件装失败。
```bash
git --version
node -v
npm -v
```
### 生成博客
从现在开始，你在windows和ubuntu下的操作几乎一样了。在网上搜帖子的时候如果是ubuntu系统下的解决方案也可以尝试在windows下解决。

新建文件夹，例如我的文件夹为： I\blog。博客相关文件将储存在此文件夹下。右键呼出gitbash。输入以下命令：
```bash
hexo init
```
如果最后出现 
>Start blogging with Hexo!

则说明生成成功。

执行以下命令
```bash
hexo g
hexo server
```
显示以下信息说明操作成功
```bash
INFO Hexo is running at http://0.0.0.0:4000/. Press Ctrl+C to stop.
```
执行完可以登录http://localhost:4000/ 查看效果。
## 博客部署
到目前为止，我们只能通过本地连接查看博客，接下来我们需要把他部署在github pages上。来，让我们登录我们上一步申请的账号。
### 创建项目代码库
点击 New 创建一个代码库。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/resp.png)

在这里需要注意仓库名必须是 用户名.github.io的形式（我这里因为已经申请了所以显示无法创建）。最后记得勾选初始化readme文件。

### 配置ssh密钥
配置好SSH密钥之后，才可以通过git实现本地代码库与github代码库同步。右键唤出gitbash进入你新建的文件夹（例如我的是I:\blog)，输入以下命令：
```bash
$ ssh-keygen -t rsa -C "your email@example.com" 
 //引号里面填写你的邮箱地址，比如我的是zhouthepassion@outlook.com
```
之后会出现：
```bash
 Generating public/private rsa key pair.  
 Enter file in which to save the key (/c/Users/you/.ssh/id_rsa):  
 //到这里可以直接回车将密钥按默认文件进行存储
```
然后会出现
```bash
 Enter passphrase (empty for no passphrase):  
 //这里是要你输入密码，其实不需要输什么密码，直接回车就行 
 Enter same passphrase again: 
```
接下来屏幕会显示

```bash
 Your identification has been saved in /c/Users/you/.ssh/id_rsa.  
 Your public key has been saved in /c/Users/you/.ssh/id_rsa.pub.  
 The key fingerprint is:  
 这里是各种字母数字组成的字符串，结尾是你的邮箱  
 The key's randomart image is:  
 这里也是各种字母数字符号组成的字符串 
```
运行以下命令,将公钥的内容复制粘贴到系统粘贴板上。
```bash
$ clip < ~/.ssh/id_rsa.pub
```
### 在github账户中添加你的公钥
点击你的github头像，进入settings，点击SSH and GPG Keys，选择New SSH key，然后把你刚才复制的公填在key那里就可以了，title可以随便填，最后点击下面的add ssh key。
### 测试
输入以下命令
```bash
$ ssh -T git@github.com
```
之后会显示
>Are you sure you want to continue connecting(yes/no)?
>

输入yes后显示
>Hi,XXXXX!You've successfully authenticated, but GitHub does not provide shell access.
>

表示设置正确。
### 配置Git个人信息
这一步相当于赋予你的电脑连接到github的权限。输入以下命令
```bash
 $ git config --global user.name "此处填你的用户名"  
 $ git config --global user.email  "此处填你的邮箱"
```
到此为止SSH Key配置成功
## 将本地hexo文件更新到GitHub仓库中
打开创建的文件夹，打开_config.yml文件（这里推荐使用Notepad++）

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/deploy.png)

拉到最后，修改deploy的属性
```bash
deploy:
  type: git
  
  repo: git@github.com:username/username.github.io.git
  branch: master
```
其中username改为你的用户名。注意冒号之后必须空一个英文空格。
在创建的文件夹中分别执行以下命令

```bash
 $ hexo g  
 $ hexo d
```
或者直接
```bash
 hexo g -d
```
执行完之后会让你输入你的Github账号和密码。如果显示以下错误，说明你的deployer没有安装成功。
```bash
ERROR Deployer not found: git
```
那就执行以下命令再安装一次:
```bash
 npm install hexo-deployer-git --save
```
再执行`hexo g -d`，你的博客就会部署到github上了。你的网址就是https://username.github.io
## 在博客上发表文章

1. 新建文章

   新建一个空文章，输入以下命令，会在项目 \source\_posts 中生成 文章标题.md 文件，文章标题根据需要命名

``` bash
$ hexo n "文章标题"
More info: [Writing](https://hexo.io/docs/writing.html)
```

2. 编辑文章

    Markdown 是 2004 年由 John Gruberis 设计和开发的纯文本格式的语法，非常的简单实用，常用的标记符号屈指可数，几分钟即可学会， .md 文件可以使用支持 Markdown 语法的编辑器编辑，我这里使用的是typora来编辑，对于初学者十分友好。这里贴出一个Markdown格式的[语法指南](https://www.jianshu.com/p/1e402922ee32)

3. 发布文章

   文章写好后，可以使用如下命令发布

```bash
 $ hexo g  
 $ hexo d
```
或者直接
```bash
 hexo g -d
```

然后就可以在刚才的网址里面看到你写的文章了


## 参考资料



搭建：https://blog.csdn.net/qq_36759224/article/details/82121420


<!--


乱码解决：https://blog.csdn.net/Aoman_Hao/article/details/79275570

;实用：https://blog.csdn.net/qq_36759224/article/details/85010191

;美化：https://blog.csdn.net/qq_36759224/article/details/85420403

;常见错误：http://www.aichengxu.com/other/2538446.htm

;next配置：http://theme-next.iissnan.com/theme-settings.htm

;关于页面：https://www.jianshu.com/p/7667d8e8f91c

;英文标签改中文改前面就行

;https://blog.csdn.net/qq_32337109/article/details/78755729只展示一部分;
;https://blog.csdn.net/lewky_liu/article/details/81277337



;gitalkhttps://asdfv1929.github.io/2018/01/20/gitalk/

;issue:https://liujunzhou.top/2018/8/10/gitalk-error/#%E6%9C%AA%E6%89%BE%E5%88%B0%E7%9B%B8%E5%85%B3%E7%9A%84Issues%E8%BF%9B%E8%A1%8C%E8%AF%84%E8%AE%BA%EF%BC%8C%E8%AF%B7%E8%81%94%E7%B3%BBXXX%E8%BF%9B%E8%A1%8C%E5%88%9B%E5%BB%BA

;分析https://marketingplatform.google.com/about/analytics/

;https://www.cnblogs.com/tengj/p/5357879.html

;```text
tags: [标签1,标签2,标签3]

```

;http://cnneillee.github.io/2017/05/10/hexo/Hexo%E8%BF%9B%E9%98%B6%E2%80%94%E2%80%94%E6%B7%BB%E5%8A%A0%E7%AB%99%E7%82%B9%E5%9C%B0%E5%9B%BE/

;sitemap

;提交http://fionat.github.io/blog/2013/10/23/sitemap/



;https://alanlee.fun/2017/12/30/google-sitemap/

;https://www.jianshu.com/p/efbeddc5eb19
;备份
https://www.simon96.online/2018/10/12/hexo-tutorial/
```
RSS：https://segmentfault.com/a/1190000012647294

https://mritd.me/2016/03/08/Hexo%E6%B7%BB%E5%8A%A0Rss%E8%AE%A2%E9%98%85/

leancloud:https://lfwen.site/2016/05/31/add-count-for-hexo-next/

https://lruihao.cn/hexo/hexo-%E6%B7%BB%E5%8A%A0%E5%9B%BE%E7%89%87%EF%BC%8C%E9%9F%B3%E4%B9%90%EF%BC%8C%E9%93%BE%E6%8E%A5%EF%BC%8C%E8%A7%86%E9%A2%91.html 网易云 图片 视频

<https://www.jianshu.com/p/2756724a5dee>图片问题终于解决了

百度推广https://www.jianshu.com/p/8c0707ce5da4

RSS<https://blog.csdn.net/u011303443/article/details/52333695>

<https://blog.tangxiaozhu.com/15250922329733.html>深度定制

<http://liuqi日历云ngwen.me/blog/2018/10/26/share-a-cute-hexo-blog-plugin-the-cloud-calendar/>

[https://tankeryang.github.io/posts/Hexo%20+%20NexT%20+%20Github%20Pages%20+%20Coding%20Pages%20+%20Gitee%20Pages%20+%20Travis%20%E5%85%A8%E6%94%BB%E7%95%A5/](https://tankeryang.github.io/posts/Hexo + NexT + Github Pages + Coding Pages + Gitee Pages + Travis 全攻略/)

<https://sesprie.bid/articles/21.html>进度条

自己要好好想想怎么实现follow效果

升级<https://11.tt/posts/2018/how-to-update-hexo-theme-next/>

后面这些还有

<https://juejin.im/post/5caddd1ff265da035e210dce#heading-49>

<https://www.jianshu.com/p/e211e9119522>

[https://whjkm.github.io/2018/07/17/Hexo%E7%89%88%E6%9C%AC%E5%8D%87%E7%BA%A7%E5%92%8CNext%E4%B8%BB%E9%A2%98%E5%8D%87%E7%BA%A7%E4%B9%8B%E5%9D%91/](https://whjkm.github.io/2018/07/17/Hexo版本升级和Next主题升级之坑/)

<https://github.com/theme-next/hexo-theme-next/blob/master/docs/zh-CN/UPDATE-FROM-5.1.X.md>

hexo-wordcounthttps://github.com/theme-next/hexo-symbols-count-time

代码复制

-->

