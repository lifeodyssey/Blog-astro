---
title: Atmospheric Correction by He
abbrlink: 23b76a18
date: 2021-07-14 15:15:26
tags:
  - Research Basis
  - Ocean Optics
categories: 学习笔记
mathjax:
copyright:
---

何老师在第二届水色遥感理论班的大气校正课程

<!-- more -->

![image-00030714150751457](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714150751457-00030714151728161.png)



![image-00030714150939841](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714150939841-00030714151728504.png)

这个框架是现在水色遥感大气校正的标准算法，计算气溶胶是使用查找表的方式，因为辐射传输计算太花时间了。查找表是针对每个卫星生成的。

![image-00030714151304935](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714151304935-00030714151728160.png)

![image-00030714151408980](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714151408980-00030714151728422.png)





![image-00030714151633347](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714151633347.png)



![image-00030714151803215](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714151803215.png)

![image-00030714151854497](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714151854497.png)

![image-00030714152038809](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714152038809.png)

这个是之前的一个做法。对于非吸收性分子和气溶胶，不需要考虑垂向变化。在标准大气压下生成，实际使用时使用辅助观测气压数据校正。

![image-00030714152608100](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714152608100.png)

瑞利散射的通用表已经做出来了，气溶胶的还比较难。

查找表在satco2上可以找到，读取的matlab代码也放在上面。紫外不实用因为光学厚度>0.4

![image-00030714152743347](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714152743347.png)

![image-00030714152843021](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714152843021.png)

![image-00030714152900905](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714152900905.png)

白沫反射贡献比较小，估算不准影响也不大，可以直接拿风速来算。$R_{wc}(\lambda)$是根据风速和经验函数算出来的。高分辨率卫星可能影响比较大，百米千米级的水色卫星影响不大。

![image-00030714153221071](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714153221071.png)

![image-00030714153409064](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714153409064.png)

最难的就是这个气溶胶的，利用black pixel假设，这幅图是一类水里面的。

![image-00030714153644456](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714153644456.png)

![image-00030714153732628](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714153732628.png)

后面这个80种气溶胶模式是用粗模态和细模态，AERONET-OC统计出来的。

![image-00030714153831140](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714153831140.png)

![image-00030714154006856](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714154006856.png)

![image-00030714154745753](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714154745753.png)

算完之后用辐射传输解出来

![image-00030714155301926](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714155301926.png)

![image-00030714155518982](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714155518982.png)

![image-00030714160101000](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714160101000.png)

被拉去帮了个忙，中间可能过了一些

![image-00030714160146582](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714160146582.png)

![image-00030714160316876](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714160316876.png)

![image-00030714160505057](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714160505057.png)

![image-00030714160733681](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714160733681.png)

![image-00030714161121970](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714161121970.png)

![image-00030714161139931](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714161139931.png)

![image-00030714161336991](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714161336991.png)

![image-00030714161424435](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714161424435.png)

![image-00030714161624885](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714161624885.png)

校正到LT上面去

![image-00030714161741793](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714161741793.png)

![image-00030714161753381](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714161753381.png)

![image-00030714161852352](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714161852352.png)

![image-00030714162146782](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714162146782.png)

![image-00030714162437954](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714162437954.png)

![image-00030714162644935](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714162644935.png)

![image-00030714170049337](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714170049337.png)

![image-00030714170106559](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714170106559.png)

![image-00030714170319914](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714170319914.png)

![image-00030714170432441](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714170432441.png)

![image-00030714170558971](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714170558971.png)

终于到了我感兴趣的地方了

![image-00030714170630435](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714170630435.png)

![image-00030714170734239](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714170734239.png)

![image-00030714171104513](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714171104513.png)

![image-00030714171343885](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714171343885.png)

![image-00030714172456970](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714172456970.png)

![image-00030714172612626](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714172612626.png)

![image-00030714172721960](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714172721960.png)

![image-00030714172913598](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714172913598.png)

![image-00030714173207910](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714173207910.png)

![image-00030714173737934](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714173737934.png)

![image-00030714173849090](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714173849090.png)

分辨率可以达到25 我去

![image-00030714174219987](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714174219987.png)

![image-00030714174249270](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714174249270.png)

![image-00030714174312920](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714174312920.png)

![image-00030714174405172](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714174405172.png)

这个应该就是Li hao的那个

![image-00030714174651469](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714174651469.png)

![image-00030714174710201](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714174710201.png)

![image-00030714174740486](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714174740486.png)

![image-00030714174848723](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714174848723.png)

![image-00030714174939288](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714174939288.png)

![image-00030714175009007](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714175009007.png)

![image-00030714175106875](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714175106875.png)

![image-00030714175136812](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image-00030714175136812.png)

就到这里就结束了。

笑死了，果然关心这个的最多。

