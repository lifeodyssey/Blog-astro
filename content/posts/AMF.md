---
title: '自动机器学习代码详解'
copyright: true
tags:
  - Automatic Machine Learning
  - Fishery Forecasting
  - Undergraduate Thesis
categories: Papers And Thesis
mathjax: true
abbrlink: 4668cbbb
date: 2019-05-16 10:22:38
---

毕业论文基于auto-sklearn做了一些自动机器学习模型的应用，模型的原理可以点击[这里](http://codewithzhangyi.com/2018/07/26/AutoML/)，我这儿讲一下我的代码怎么用，相当于是一个详细版的注释。

代码地址见<https://github.com/lifeodyssey/AMF>。

<!-- more -->

进入到服务器amf目录下，输入

```bash
source py3/bin/activate
```

之后每一行的开头会显示(py3)字样，即代表进入虚拟环境。

之后直接输入

```bash
python code.py
```

即可开始运行程序。

以下是程序的详细说明，如果要修改代码，可以在服务器中直接使用vim进行修改，也可以在本地修改好之后用PSFTP推送到服务器上。

```python
import autosklearn.regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#前面这四行是引入相应的环境包
rawdata=pd.read_excel('rawdata.xlsx')#单引号输入要读入文件的路径，文件的格式参考服务器中的rawdata.xlsx
Y=rawdata[['cpue']]#单引号内写要预测的变量名

Y=np.log10(Y+1)#这里采用了对数变换，可以去掉
X=rawdata[['lon','lat','sst','chla','doy']]#单引号内输入要使用的变量名
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=7)#test size代表了测试集在整个数据集中的占比，这里选取的是0.3即30%


automl = autosklearn.regression.AutoSklearnRegressor(
    include_estimators=["random_forest","decision_tree","gradient_boosting","xgradient_boosting"],
    #这里只放了几个我认为效果比较好的模型，模型种类参见https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regression，如果想换用模型，把双引号内的名称更改即可；如果想删除或者添加，直接去掉相应的名称即可
    exclude_estimators=None,
    include_preprocessors=["no_preprocessing", ],
    exclude_preprocessors=None,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 10},#这里选了十折交叉验证来确定最优参数
    )
automl.fit(x_train, y_train.values.ravel())

automl.sprint_statistics()
automl.show_models()
automl.refit(x_train, y_train.values.ravel())
#如果想输入新的自变量来做预测的话,将下面几句前面的#去除即可
#predata=pd.read_excel('rawdata.xlsx')#单引号内输入文件路径
#X=predata[['lon','lat','sst','chla','doy']]#单引号内输入选取的变量名
y_pre = automl.predict(X)
ypre=np.power(10,ypre)-1#变换回来
ypre=pd.DataFrame(ypre)
result=pd.concat([rawdata,ypre])#如果输入新的自变量来做预测，将rawdata改为predata
result.to_excel('result.xlsx')


```

