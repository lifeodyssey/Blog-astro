---
title: 'AutoML Code Explained'
copyright: true
tags:
  - Automatic Machine Learning
  - Fishery Forecasting
  - Undergraduate Thesis
categories: Papers And Thesis
mathjax: true
abbrlink: 4668cbbb
slug: automl-code-explained
date: 2019-05-16 10:22:38
lang: en
---

My undergraduate thesis applied some automatic machine learning models based on auto-sklearn. For the model principles, you can click [here](http://codewithzhangyi.com/2018/07/26/AutoML/). Here I'll explain how to use my code, essentially a detailed version of the comments.

Code repository: <https://github.com/lifeodyssey/AMF>.

<!-- more -->

Navigate to the amf directory on the server and enter:

```bash
source py3/bin/activate
```

After this, each line will show (py3) at the beginning, indicating you've entered the virtual environment.

Then simply enter:

```bash
python code.py
```

to start running the program.

Below is a detailed explanation of the program. If you want to modify the code, you can use vim directly on the server, or modify it locally and push it to the server using PSFTP.

```python
import autosklearn.regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# The first four lines import the required packages
rawdata=pd.read_excel('rawdata.xlsx')  # Enter the file path in quotes, refer to rawdata.xlsx on the server for file format
Y=rawdata[['cpue']]  # Enter the variable name to predict in quotes

Y=np.log10(Y+1)  # Log transformation applied here, can be removed
X=rawdata[['lon','lat','sst','chla','doy']]  # Enter the variable names to use in quotes
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=7)  # test_size represents the proportion of test set, here 0.3 means 30%
```

```python
automl = autosklearn.regression.AutoSklearnRegressor(
    include_estimators=["random_forest","decision_tree","gradient_boosting","xgradient_boosting"],
    # Only a few models I found effective are included here
    # For model types, see https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regression
    # To change models, modify the names in quotes; to add or remove, simply adjust the list
    exclude_estimators=None,
    include_preprocessors=["no_preprocessing", ],
    exclude_preprocessors=None,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 10},  # 10-fold cross-validation for optimal parameters
    )
automl.fit(x_train, y_train.values.ravel())
```

```python
automl.sprint_statistics()
automl.show_models()
automl.refit(x_train, y_train.values.ravel())
# To make predictions with new independent variables, remove the # from the following lines
#predata=pd.read_excel('rawdata.xlsx')  # Enter file path in quotes
#X=predata[['lon','lat','sst','chla','doy']]  # Enter selected variable names in quotes
y_pre = automl.predict(X)
ypre=np.power(10,ypre)-1  # Transform back
ypre=pd.DataFrame(ypre)
result=pd.concat([rawdata,ypre])  # If using new independent variables, change rawdata to predata
result.to_excel('result.xlsx')
```
