---
title: pandas基础
tags:
  - Research
  - python
categories: 学习笔记
mathjax: true
abbrlink: 50bcdc38
date: 2020-09-09 09:13:50
copyright:
---

其实我一点儿都不想用pandas的dataframe，总感觉这个东西很难用，也许是我的水平还没达到。但是好多时候读写实验室的excel和输出成excel又必须用这个，之前用的seaborn也是基于pandas的，那就再把基础操作复习一下。

<!-- more -->

# 读写与创建

## 数据读写

### sql

- **读取**

```python
import pandas as pd
from sqlalchemy import create_engine
db = create_engine("mysql+pymysql://用户名:用户密码@localhost:端口号（3306）/使用的数据库名?charset=utf8")
sql = "select * from text"
df = pd.read_sql(sql, db, index_col="index") # index_col设置索引列，默认自动生成索引
```

- **写入**

```python
sql.to_sql(df, name='test', con=db,
                 if_exists="append",# 如果表存在：append追加 replace删除原表新建并插入 fail不插入
                 index=False # 设置df的索引不插入数据库
                 )
```

### excel

- **读取**

```python
df = pd.read_excel(r'file_path',
                   sheet_name='指定sheet,默认第一个',
                   index=False, # 不读取excel中的索引，自动生成新索引
                   index_col=0, # 将第0列设置为索引
                   header=0, # 将第n行设置为columns, 默认是0，可以设置为None（自动生成0-n的columns）
                   usecols=[0, 2] # 只导入0, 2列
                   )
```

- **写入**

```python
'''
按照不同sheet写入
'''
# 创建表格
excelWriter = pd.ExcelFile('file_path/test.xlsx')
# 写入表格
df.to_excel(
    excelWriter,
    sheet_name='',
    insex=False, # 设置df的索引不传入excel
    encoding='utf-8',
    columns=['a', 'b'], # 指定某列写入excel
    na_rep=0, # 缺失值处理（填充为0）
    inf_rep=0, # 无穷值处理（填充为0）
)
# 保存（不保存不生效）
excelWriter.save()
'''
直接写入
'''
df.to_excel('file_path/test.xlsx') # 参数：insex、encoding、columns、na_rep、inf_rep
```

### csv

- **读取**

```python
df = pd.read(
    r'file_path/test.csv',
    sep="", # 指定分隔符，默认是逗号
    nrows=2, # 指定读取行数
    encoding='utf-8',
    engine='python', # 当路径存在中文会报错，加上这个即解决
    usecols=[0, 2], # 仅导入0, 2列
    index_col=0, # 将第0列设置为索引
    header=0 # 将第n行设置为columns, 默认是0，可以设置为None（自动生成0-n的columns）
)
```

- **写入**

```python
df.to_csv(
    r'file_path/test.csv',
    index=False, # 索引列不写入
    columns=['a', 'b'], # 指定写入的列
    sep=',', # 设置分隔符（默认是逗号）
    na_rep=0, # 缺失值填充为0
    encoding='utf-8',
    #inf_rep=0 没有这个参数
)
```

### txt

- **读取**

```python
pd.read_table(r'file_path/test.txt', sep='') #也可以用来读取csv文件
```

- **写入**

```python
df.to_csv(
    r'file_path/test.csv',
    index=False, # 索引列不写入
    columns=['a', 'b'], # 指定写入的列
    sep=',', # 设置分隔符（默认是逗号）
    na_rep=0, # 缺失值填充为0
    encoding='utf-8',
    #inf_rep=0 没有这个参数
)
```

来源https://www.modb.pro/db/26894

## 创建

最让我不爽的就是dataframe没有像np.zeros,np.ones这种根据已有的dataframe来初始化一个空dataframe.

```python
df_empty=pd.Dataframe(columns=['A','B','C','D'])
```

所以有一种办法就是把已有的datafram列名提取出来，然后再去创建。

# 索引

Pandas 数据的索引就像一本书的目录，让我们很快地找到想要看的章节，作为大量数据，创建合理的具有业务意义的索引对我们分析数据至关重要。

## 认识索引

下图是一个简单的 DataFrame 中索引的示例：

![pandas index](https://www.gairuo.com/file/pic/2020/04/pandas_index_01.jpg)

其中：

- 行索引是数据的索引，列索引指向的是一个 Series
- DataFrame 的索引也是系列形成的 Series 的索引
- 建立索引让数据更加直观明确，如每行数据是针对一个国家的
- 建立索引方便数据处理
- 索引允许重复，但业务上一般不会让它重复

有时一个行和列层级较多的数据会出现[多层索引](https://www.gairuo.com/p/pandas-multiIndex) 的情况。

## 建立索引

之前我们学习了加载数据生成 DataFrame 时可以指定索引

```python
data = 'https://www.gairuo.com/file/data/dataset/team.xlsx'
df = pd.read_excel(data, index_col='name') # 设置索引为 name
df
'''
      team  Q1  Q2  Q3  Q4
name
Liver    E  89  21  24  64
Arry     C  36  37  37  57
Ack      A  57  60  18  84
Eorge    C  93  96  71  78
Oah      D  65  49  61  86
'''
```

如果加载时没有指定索引，我们可以使用 `df.set_index()` 指定：

```python
df.set_index('month') # 设置月份为索引
df.set_index(['month', 'year']) # 设置月份和年为多层索引
'''
            sale
month year
1     2012    55
4     2014    40
      2013    84
10    2014    31
'''

s = pd.Series([1, 2, 3, 4])
df.set_index(s) # 指定一个索引
df.set_index([s, 'year']) # 指定的索引和现有字段同时指定
df.set_index([s, s**2]) # 计算索引

# 其他的参数
df.set_index('month', drop=False) # 保留原列
df.set_index('month', append=True) # 保留原来的索引
df.set_index('month', inplace=True) # 建立索引并重写覆盖 df
```

## 重置索引

有时我们想取消已有的索引，以重新来过，可以使用 `df.reset_index()`：

```python
df.reset_index() # 清除索引
df.set_index('month').reset_index() # 相当于啥也没干
# 删除原索引，month 列没了
df.set_index('month').reset_index(drop=True)
df2.reset_index(inplace=True) # 覆盖使生效
# year 一级索引取消
df.set_index(['month', 'year']).reset_index(level=1)
df2.reset_index(level='class') # 同上使用层级索引名
df.reset_index(level='class', col_level=1) # 列索引
# 不存在层级名称的填入指定名称
df.reset_index(level='class', col_level=1, col_fill='species')
```

## 索引类型

为了适应各种业务数据的处理，索引又针对各种类型数据定义了不同的索引类型：

### 数字索引 Numeric Index

共有以下几种：

- RangeIndex: 单调整数范围的不可变索引。
- Int64Index: int64类型，有序可切片集合的不可变ndarray。
- UInt64Index: 无符号整数标签的
- Float64Index: Float64 类型

```python
pd.RangeIndex(1,100,2)
# RangeIndex(start=1, stop=100, step=2)
pd.Int64Index([1,2,3,-4], name='num')
# Int64Index([1, 2, 3, -4], dtype='int64', name='num')
pd.UInt64Index([1,2,3,4])
# UInt64Index([1, 2, 3, 4], dtype='uint64')
pd.Float64Index([1.2,2.3,3,4])
# Float64Index([1.2, 2.3, 3.0, 4.0], dtype='float64')
```

### 类别索引 CategoricalIndex

类别只能包含有限数量的（通常是固定的）可能值（类别）。 可以理解成枚举，比如性别只有男女，但在数据中每行都有，如果按文本处理会效率不高。类别的底层是 pandas.Categorical。

```python
pd.CategoricalIndex(['a', 'b', 'a', 'b'])
# CategoricalIndex(['a', 'b', 'a', 'b'], categories=['a', 'b'], ordered=False, dtype='category')
```

类别后边后有专门的讲解，只有在体量非常大的数据面前才能显示其优势。

### 间隔索引 IntervalIndex

```python
pd.interval_range(start=0, end=5)
'''
IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
              closed='right',
              dtype='interval[int64]')
'''
```

### 多层索引 MultiIndex

教程后边会有专门的讲解。

```python
arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
'''
MultiIndex([(1,  'red'),
            (1, 'blue'),
            (2,  'red'),
            (2, 'blue')],
           names=['number', 'color'])
'''
```

### 时间索引 DatetimeIndex

```python
# 从一个日期连续到另一个日期
pd.date_range(start='1/1/2018', end='1/08/2018')
# 指定开始时间和周期
pd.date_range(start='1/1/2018', periods=8)
# 以月为周期
pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
# 周期嵌套
pd.period_range(start=pd.Period('2017Q1', freq='Q'),
                end=pd.Period('2017Q2', freq='Q'), freq='M')
```

### 时间差 TimedeltaIndex

```python
pd.TimedeltaIndex(data =['06:05:01.000030', '+23:59:59.999999',
                         '22 day 2 min 3us 10ns', '+23:29:59.999999',
                         '+12:19:59.999999'])
# 使用 datetime
pd.TimedeltaIndex(['1 days', '1 days, 00:00:05',
                   np.timedelta64(2, 'D'),
                   datetime.timedelta(days=2, seconds=2)])
```

### 周期索引 PeriodIndex

```python
t = pd.period_range('2020-5-1 10:00:05', periods=8, freq='S')
pd.PeriodIndex(t,freq='S')
```

## 索引对象

行和列的索引在 Pandas 里其实是一个 `Index` 对象，以下是创建一个 `index` 对象的方法：

### 创建对象

```python
pd.Index([1, 2, 3])
# Int64Index([1, 2, 3], dtype='int64')
pd.Index(list('abc'))
# Index(['a', 'b', 'c'], dtype='object')
# 可以定义一相 name
pd.Index(['e', 'd', 'a', 'b'], name='something')
```

### 查看

```python
df.index
# RangeIndex(start=0, stop=4, step=1)
df.columns
# Index(['month', 'year', 'sale'], dtype='object')
```

### 属性

以下方法也适用于 `df.columns`, 因为都是 index 对象：

```python
# 属性
df.index.name # 名称
df.index.array # array 数组
df.index.dtype # 数据类型
df.index.shape # 形状
df.index.size # 元素数量
df.index.values # array 数组
# 其他，不常用
df.index.empty # 是否为空
df.index.is_unique # 是否不重复
df.index.names # 名称列表
df.index.is_all_dates # 是否全是日期时间
df.index.has_duplicates # 是否有重复值
df.index.values # 索引的值 array
```

### 操作

以下方法也适用于 `df.columns`, 因为都是 index 对象：

```python
# 方法
df.index.astype('int64') # 转换类型
df.index.isin() # 是否存在，见下方示例
df.index.rename('number') # 修改索引名称
df.index.nunique() # 不重复值的数量
df.index.sort_values(ascending=False,) # 排序,倒序
df.index.map(lambda x:x+'_') # map 函数处理
df.index.str.replace('_', '') # str 替换
df.index.str.split('_') # 分隔
df.index.to_list() # 转为列表
df.index.to_frame(index=False, name='a') # 转成 DataFrame
df.index.to_series() # 转 series
df.index.to_numpy() # 转为 numpy
df.index.unique() # 去重
df.index.value_counts() # 去重及数量
df.index.where(df.index=='a') # 筛选
df.index.rename('grade', inplace=False) # 重命名索引名称
df.index.rename(['species', 'year']) # 多层，重命名索引名称
df.index.max() # 最大值
df.index.argmax() # 最大索引值
df.index.any()
df.index.all()
df.index.T # 转置，多层索引里很有用

# 其他，不常用
df.index.append(pd.Index([4,5])) # 追加
df.index.repeat(2) # 重复几次
df.index.inferred_type # 推测数据类型
df.index.hasnans # 有没有空值
df.index.is_monotonic_decreasing # 是否单调递减
df.index.is_monotonic # 是否单调递增
df.index.is_monotonic_increasing # 是否单调递增
df.index.nbytes # 基础数据中的字节数
df.index.ndim # 维度数，维数
df.index.nlevels # 索引层级数，通常为 1
df.index.min() # 最小值
df.index.argmin() # 最小索引值
df.index.argsort() # 顺序值组成的数组
df.index.asof(2) # 返回最近的索引
# numpy dtype or pandas type
df.index.astype('int64', copy=True) # 深拷贝
# 拷贝
df.index.copy(name='new', deep=True, dtype='int64')
df.index.delete(1) # 删除指定位置
# 对比不同
df.index.difference(pd.Index([1,2,4]), sort=False)
df.index.drop('a', errors='ignore') # 删除
df.index.drop_duplicates(keep='first') # 去重值
df.index.droplevel(0) # 删除层级
df.index.dropna(how='all') # 删除空值
df.index.duplicated(keep='first') # 重复值在结果数组中为True
df.index.equals(df.index) # 与另外一个索引对象是否相同
df.index.factorize() # 分解成（array:0-n, Index）
df.index.fillna(0, {0:'nan'}) # 填充空值
# 字符列表, 把 name 值加在第一位, 每个值加10
df.index.format(name=True, formatter=lambda x:x+10)

# 返回一个 array, 指定值的索引位数组，不在的为 -1
df.index.get_indexer([2,9])
# 获取 指定层级 Index 对象
df.index.get_level_values(0)
# 指定索引的位置，见示例
df.index.get_loc('b')
df.index.insert(2, 'f') # 在索引位 2 插入 f
df.index.intersection(df.index) # 交集
df.index.is_(df.index) # 类似 is 检查
df.index.is_categorical() # 是否分类数据
df.index.is_type_compatible(df.index) # 类型是否兼容
df.index.is_type_compatible(1) # 类型是否兼容

df.index.isna() # array 是否为空
df.index.isnull() # array 是否缺失值
df.index.join(df.index, how='left') # 连接
df.index.notna() # 是否不存在的值
df.index.notnull() # 是否不存在的值
df.index.ravel() # 展平值的ndarray
df.index.reindex(['a','b']) # 新索引 (Index,array:0-n)
df.index.searchsorted('f') # 如果插入这个值排序后在哪个索引位
df.index.searchsorted([0, 4]) # array([0, 3]) 多个
df.index.set_names('quarter') # 设置索引名称
df.index.set_names('species', level=0)
df.index.set_names(['kind', 'year'], inplace=True)
df.index.shift(10, freq='D') # 日期索引向前移动 10 天
idx1.symmetric_difference(idx2) # 两个索引不同的内容
idx1.union(idx2) # 拼接

df.add_prefix('t_') # 表头加前缀
df.add_suffix('_d') # 表头加后缀
df.first_valid_index() # 第一个有值的索引
df.last_valid_index() # 最后一个有值的索引
```

## 索引重命名

对行和列的索引名进行修改。

```python
# 一一对应修改列索引
df.rename(columns={"A": "a", "B": "c"})
df.rename(str.lower, axis='columns')
# 修改行索引
df.rename(index={0: "x", 1: "y", 2: "z"})
df.rename({1: 2, 2: 4}, axis='index')
# 修改数据类型
df.rename(index=str)
# 重新修改索引
replacements = {l1:l2 for l1, l2 in zip(list1, list2)}
df.rename(replacements)
```

## 索引名重命名

注意，这是修改索引的名称，不是索引或者列名本身：

```python
s.rename_axis("animal") # 索引重命名
df.rename_axis(["dow", "hr"]) # 多层索引索引名修改
df.rename_axis('info', axis="columns") # 修改行索引名
# 修改多层列索引名
df.rename_axis(index={'a': 'A', 'b': 'B'})
# 修改多层列行索引名
df.rename_axis(columns={'name': 's_name', 'b': 'B'})
df.rename_axis(columns=str.upper) # 行索引名变大写
```

## 部分示例

```python
# idx.isin() 是否存在
idx = pd.Index([1,2,3])
df.index.isin(idx)
# array([False, False, False, False])
df.index.isin(['a','b'])
# array([ True,  True, False, False])
midx = pd.MultiIndex.from_arrays([[1,2,3],
                                 ['red', 'blue', 'green']],
                                 names=('number', 'color'))
midx.isin([(1, 'red'), (3, 'red')])
# array([ True, False, False])
dates = ['2000-03-11', '2000-03-12', '2000-03-13']
dti = pd.to_datetime(dates)
dti.isin(['2000-03-11'])
# array([ True, False, False])

# i.argsort() 排序
# 将对索引进行排序的整数索引，见下文示例
idx = pd.Index(['b', 'a', 'd', 'c'])
order = idx.argsort() # array([1, 0, 3, 2])
idx[order] # Index(['a', 'b', 'c', 'd'], dtype='object')

# i.asof(2) 返回最近的索引, 支持日期，可实现找最近日期
# 从索引中返回标签；如果不存在，则返回前一个标签
idx2 = pd.Index([1,3,6])
idx2.asof(5) # 3
idx2.asof(6) # 5
idx2.asof(-1) # nan

# index.get_loc 指定索引的位置，见示例
unique_index = pd.Index(list('abc'))
unique_index.get_loc('b') # 1
monotonic_index = pd.Index(list('abbc'))
monotonic_index.get_loc('b') # slice(1, 3, None)
non_monotonic_index = pd.Index(list('abcb'))
non_monotonic_index.get_loc('b')
# array([False,  True, False,  True], dtype=bool)
```

# 查询与修改

## 数据检查

我们一拿到数据需要对数据有一个抽查，一方面是了解数据结构，另一方面随机检查一下数据的质量问题。常用的：

| 语法           | 操作                         | 返回结果  |
| -------------- | ---------------------------- | --------- |
| `df.head(n)`   | 查看 DataFrame 对象的前n行   | DataFrame |
| `df.tail(n)`   | 查看 DataFrame 对象的最后n行 | DataFrame |
| `df.sample(n)` | 查看 n 个样本，随机          | DataFrame |

以上都是选择整行。

### 查看头部 df.head()

每次加载数据后一般需要看一下头部数据

```python
df.head()
out:
    name team  Q1  Q2  Q3  Q4
0  Liver    E  89  21  24  64
1   Arry    C  36  37  37  57
2    Ack    A  57  60  18  84
3  Eorge    C  93  96  71  78
4    Oah    D  65  49  61  86

# 可指定数量
df.head(15)
```

### 查看尾部 df.tail()

查看最后的尾部数据。

```python
df.head()
out:
        name team  Q1  Q2  Q3  Q4
95   Gabriel    C  48  59  87  74
96   Austin7    C  21  31  30  43
97  Lincoln4    C  98  93   1  20
98       Eli    E  11  74  58  91
99       Ben    E  21  43  41  74

# 可指定数量
df.tail(15)
```

### 查看样本 df.sample()

`df.sample()` 会随机返回一条样本数据。

```python
df.sample()
out:
     name team  Q1  Q2  Q3  Q4
79  Tyler    A  75  16  44  63

# 可指定数量
df.sample(15)
```

数据截取：

```python
# 去掉索引之前和之后的数据
df.truncate(before=2, after=4) # 只要索引 2-4
s.truncate(before="60", after="66")
df.truncate(before="A", after="B", axis="columns") # 选取列
```

## 操作列

以下两种方法都可以代表一列：

```python
df['name'] # 会返回本列的 Series
df.name
df.Q1
# df.1Q 即使列名叫 1Q 也无法使用
# df.my name 有空格也无法调用，可以处理加上下划线
```

注意，当列名为一个合法的 python 变量时可以直接作为属性去使用。

## 选择行列部分

有时我们需要按条件选择部分列、部分行，一般常用的有：

| 操作             | 语法            | 返回结果  |
| ---------------- | --------------- | --------- |
| 选择列           | `df[col]`       | Series    |
| 按索引选择行     | `df.loc[label]` | Series    |
| 按数字索引选择行 | `df.iloc[loc]`  | Series    |
| 使用切片选择行   | `df[5:10]`      | DataFrame |
| 用表达式筛选行   | `df[bool_vec]`  | DataFrame |

接下来我们将重点介绍一下这些查询的方法。

## 切片 []

我们可以像列表那样利用切片功能选择部分行的数据，但是不支持索引一条：

```python
df[:2] # 前两行数据
df[4:10]
df[:] # 所有数据，一般没这么用的
df[:10:2] # 按步长取
s[::-1] # 反转顺序
```

也可以选择列：

```python
df['name'] # 只要一列，Series
df[['Q1', 'Q2']] # 选择两列
df[['name']] # 选择一列，返回 DataFrame，注意和上例区别
```

## 按标签 .loc

`df.loc()` 的格式为 df.loc[<索引表达式>, <列表达式>]，表达式支持以下形式：

单个标签:

```python
# 代表索引，如果是字符需要加引号
df.loc[0] # 选择索引为 0 的行
df.loc[8]
```

单个列表标签：

```python
df.loc[[0,5,10]] # 指定索引 0，5，10 的行
df.loc[['Eli', 'Ben']] # 如果索引是 name
# 真假选择，长度要和索引一样
df.loc[[False, True]*50] # 为真的列显示，隔一个显示一个
```

带标签的切片（包括起始和停止）：

```python
df.loc[0:5] # 索引切片, 代表0-5行，包括5
df.loc['2010':'2014'] # 如果索引是时间可以用字符查询
df.loc[:] # 所有
# 本方法支持 Series
```

列筛选，必须有行筛选：

```python
dft.loc[:, ['Q1', 'Q2']] # 所有行，Q1 和 Q2两列
dft.loc[:, ['Q1', 'Q2']] # 所有行，Q1 和 Q2两列
dft.loc[:10, 'Q1':] # 0-10 行，Q1后边的所有列
```

## 按位置 .iloc

`df.iloc` 与 `df.loc` 相似，但只能用自然索引（行和列的 0 - n 索引），不能用标签。

```python
df.iloc[:3]
df.iloc[:]
df.iloc[2:20:3]
s.iloc[:3]
```

## 取具体值 .at

类似于 loc, 但仅取一个具体的值，结构为 at[<索引>,<列名>]：

```python
# 注：索引是字符需要加引号
df.at[4, 'Q1'] # 65
df.at['lily', 'Q1'] # 65 假定索引是 name
df.at[0, 'name'] # 'Liver'
df.loc[0].at['name'] # 'Liver'
# 指定列的值对应其他列的值
df.set_index('name').at['Eorge', 'team'] # 'C'
df.set_index('name').team.at['Eorge'] # 'C'
# 指定列的对应索引的值
df.team.at[3] # 'C'
```

同样 iat 和 iloc 一样，仅支持数字索引：

```python
df.iat[4, 2] # 65
df.loc[0].iat[1] # 'E'
```

.get 可以做类似字典的操作，如果无值给返回默认值（例中是0）：

```python
df.get('name', 0) # 是 name 列
df.get('nameXXX', 0) # 0, 返回默认值
s.get(3, 0) # 93, Series 传索引返回具体值
df.name.get(99, 0) # 'Ben'
```

## 表达式筛选

`[]` 切片里可以使用表达式进行筛选：

```python
df[df['Q1'] == 8] # Q1 等于8
df[~(df['Q1'] == 8)] # 不等于8
df[df.name == 'Ben'] # 姓名为Ben
df.loc[df['Q1'] > 90, 'Q1':]  # Q1 大于90，只显示 Q1
df.loc[(df.Q1 > 80) & (df.Q2 < 15)] # and 关系
df.loc[(df.Q1 > 90) | (df.Q2 < 90)] # or 关系
df[df.Q1 > df.Q2]
```

`df.loc` 里的索引部分可以使用表达式进行数据筛选。

```python
df.loc[df['Q1'] == 8] # 等于8
df.loc[df.Q1 == 8] # 等于8
df.loc[df['Q1'] > 90, 'Q1':] # Q1 大于90，只显示 Q1
# 其他表达式与切片一致

df.loc[:, lambda df: df.columns.str.len()==4] # 真假组成的序列
df.loc[:, lambda df: [i for i in df.columns if 'Q' in i]] # 列名列表
df.iloc[:3, lambda df: df.columns.str.len()==2] # 真假组成的序列
```

逻辑判断和函数：

```python
df.eq() # 等于相等 ==
df.ne() # 不等于 !=
df.le() # 小于等于 >=
df.lt() # 小于 <
df.ge() # 大于等于 >=
df.gt() # 大于 >
# 都支持  axis{0 or ‘index’, 1 or ‘columns’}, default ‘columns’
df[df.Q1.ne(89)] # Q1 不等于8
df.loc[df.Q1.gt(90) & df.Q2.lt(90)] # and 关系 Q1>90 Q2<90
```

其他函数：

```python
# isin
df[df.team.isin(['A','B'])] # 包含 AB 两组的
df[df.isin({'team': ['C', 'D'], 'Q1':[36,93]})] # 复杂查询，其他值为 NaN
```

## 函数筛选

```python
df[lambda df: df['Q1'] == 8] # Q1为8的
df.loc[lambda df: df.Q1 == 8, 'Q1':'Q2'] # Q1为8的, 显示 Q1 Q2
```

## where 和 mask

```python
s.where(s > 90) # 不符合条件的为 NaN
s.where(s > 90, 0) # 不符合条件的为 0
# np.where, 大于80是真否则是假
np.where(s>80, True, False)
np.where(df.num>=60, '合格', '不合格')

s.mask(s > 90) # 符合条件的为 NaN
s.mask(s > 90, 0) # 符合条件的为 0

# 例：能被整除的显示，不能的显示相反数
m = df.loc[:,'Q1':'Q4'] % 3 == 0
df.loc[:,'Q1':'Q4'].where(m, -df.loc[:,'Q1':'Q4'])

# 行列相同数量，返回一个 array
df.lookup([1,3,4], ['Q1','Q2','Q3']) # array([36, 96, 61])
df.lookup([1], ['Q1']) # array([36])
```

## query

```python
df.query('Q1 > Q2 > 90') # 直接写类型 sql where 语句
df.query('Q1 + Q2 > 180')
df.query('Q1 == Q2')
df.query('(Q1<50) & (Q2>40) and (Q3>90)')
df.query('Q1 > Q2 > Q3 > Q4')
df.query('team != "C"')
df.query('team not in ("E","A","B")')
# 对于名称中带有空格的列，可以使用反引号引起来
df.query('B == `team name`')

# 支持传入变量，如：大于平均分40分的
a = df.Q1.mean()
df.query('Q1 > @a+40')
df.query('Q1 > `Q2`+@a')

# df.eval() 用法与 df.query 类似
df[df.eval("Q1 > 90 > Q3 > 10")]
df[df.eval("Q1 > `Q2`+@a")]
```

## filter

使用 filter 可以对行名和列名进行筛选。

```python
df.filter(items=['Q1', 'Q2']) # 选择两列
df.filter(regex='Q', axis=1) # 列名包含Q的
df.filter(regex='e$', axis=1) # 以 e 结尾的
df.filter(regex='1$', axis=0) # 正则, 索引名包含1的
df.filter(like='2', axis=0) # 索引中有2的
# 索引中2开头列名有Q的
df.filter(regex='^2', axis=0).filter(like='Q', axis=1)
```

上面两个来自于https://www.gairuo.com/

# 合并与新增行列

## 新增列

假设原始数据如下：

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'num_legs': [4, 2], 'num_wings': [0, 2]},
                  index=['dog', 'hawk'])
slen = len(df['num_legs'])
```



1）直接赋值

```python
df['a'] = pd.Series(np.random.randn(slen), index=df.index) # index要记得添加
df['b'] = None # 添加一列值为None
df['c'] = [2, 4] # 添加列表数据

# c1和c3列的顺序是一样的， c2则与之相反，具体看下文
df['c1'] = ['no', 'yes']
df.index = [1, 0]
df['c2'] = pd.Series(['no', 'yes'])
df['c3'] = pd.Series(['no', 'yes'], index=df.index)
```

2）loc方法

```python
df.loc[:,'d'] = pd.Series(np.random.randn(slen), index=df.index)
df.loc[:, 'd'] = [2, 4]
```

3）insert方法

*insert方法使用的列名不能有重复值，连更新都不能*

```python
df.insert(len(df.columns), 'e', pd.Series(np.random.randn(slen)), index=df.index)
df.insert(len(df.columns), 'ee', [1,2])
```

4）assign方法

assign方法参数可以是Series、标量、列表，还可以同时添加多列

```python
df = df.assign(f=df.num_wings.mean())  # 将num_wings这列的平均值作为新增列f的结果
df = df.assign(A=df.num_wings.sum(), B=[1,2])  # 新增列A和B
```

5）concat方法

```python
pd.concat([df, pd.Series(['yes', 'yes']).rename('t')], axis=1) # 增加列t
```

注意点：

- 每个方法的参数都可以是Series、标量、列表
- insert方法中新增的列名不能跟已有的一样，即使更新刚刚新增的列也会出错
- `df['a']=pd.Series(['no', 'yes']`的index顺序如果被修改，默认是以Series的index为准，可以通过`index=df.index`来指定按照原始DataFrame的index顺序

## 新增行

```python
import pandas as pd
import numpy as np

# 创建空白DataFrame
df = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])
```

1）使用loc

```python
for i in range(4):
    df.loc[i] = [np.random.randint(-1, 1) for n in range(3)]
    # df.loc[i] = 5 添加一条数据都为5的记录
```



2）使用append

```python
df.append({'lib': 2, 'qty1': 3, 'qty2': 4}, ignore_index=True) 

# append也可以直接添加DataFrame
df2 = pd.DataFrame([[1,2,3], [2,3,4]], columns=['lib', 'qty1', 'qty2'])
df.append(df2, ignore_index=True)  # ignore_index设置为True，index将会忽略df2的index
```

3）重新生成DataFrame

循环将要添加的数据以字典的形式保存到一个列表中，在用列表创建出DataFrame

```python
row_list = [] 
input_rows = [[1,2,3], [2,3,4]] # 待插入数据
for row in input_rows:
    dict1 = dict(lib=row[0], qty1=row[1], qty2=row[2]) # 将数据转为字典
    row_list.append(dict1) # 保存到列表中
df = pd.DataFrame(row_list)
```

以上两个来源于https://amberwest.github.io/

## 合并

### append() Vs. concat()

连接或者合并DataFrame的时候一般有两种方式：纵向和横向。听起来总是觉得有点迷迷糊糊的。通俗的解释就是，纵向就是把两个或多个DataFrame纵向（从上到下）连接到一个DataFrame当中，index和column有重复情况也不进行任何操作，就是粗暴的纵向拼接DataFrame。横向就是会考虑如果有相同的index的话就会把相同index上所有列的数据合并在一起了，简单点理解就是相当于使用Excel中的V-lookup在两张有相同id但不同数据的表中进行了数据的融合。连接与合并DataFrame的常用函数有两个`append()`,`concat()`还有`merge()`。其中append()只能进行纵向连接，而`concat()`和`merge()`可以进行both。`concat()`默认是进行纵向连接，也就是跟`append()`效果一样，如果想要使用`concat()`进行横向合并则需要在`concat()`中声明变量axis。默认值：`concat(axis=0)`纵向连接，`concat(axis=1)`横向合并。下面举几个例子：

```python
In [1]: population = pd.read_csv('population_00.csv', index_col=0)
In [1]: unemployment = pd.read_csv('unemployment_00.csv', index_col=0)
In [1]: print(population)
|               | 2010 Census Population |
|---------------|------------------------|
| Zip Code ZCTA |                        |
| 57538         | 322                    |
| 59916         | 130                    |
| 37660         | 40038                  |
| 2860          | 45199                  |
In [4]: print(unemployment)
|       | unemployment | participants |
|-------|--------------|--------------|
| Zip   |              |              |
| 2860  | 0.11         | 34447        |
| 46167 | 0.02         | 4800         |
| 1097  | 0.33         | 42           |
| 80808 | 0.07         | 4310         |
```

以上为两个数据文件中数据的情况，下面讲举例说明append()和concat(axis=0)默认值对DataFrame纵向连接的结果，两种方式得到的结果是完全相同的：

```python
In [5]: population.append(unemployment)
Out[5]:
|       | 2010 Census Population participants unemployment | participants | unemployment |
|-------|--------------------------------------------------|--------------|--------------|
| 57538 | 322.0                                            | NaN          | NaN          |
| 59916 | 130.0                                            | NaN          | NaN          |
| 37660 | 40038.0                                          | NaN          | NaN          |
| 2860  | 45199.0                                          | NaN          | NaN          |
| 2860  | NaN                                              | 34447.0      | 0.11         |
| 46167 | NaN                                              | 4800.0       | 0.02         |
| 1097  | NaN                                              | 42.0         | 0.33         |
| 80808 | NaN                                              | 4310.0       | 0.07         |

In [6]: pd.concat([population, unemployment], axis=0)
Out[6]:
|       | 2010 Census Population participants unemployment | participants | unemployment |
|-------|--------------------------------------------------|--------------|--------------|
| 57538 | 322.0                                            | NaN          | NaN          |
| 59916 | 130.0                                            | NaN          | NaN          |
| 37660 | 40038.0                                          | NaN          | NaN          |
| 2860  | 45199.0                                          | NaN          | NaN          |
| 2860  | NaN                                              | 34447.0      | 0.11         |
| 46167 | NaN                                              | 4800.0       | 0.02         |
| 1097  | NaN                                              | 42.0         | 0.33         |
| 80808 | NaN                                              | 4310.0       | 0.07         |
```

这里我们可以看到zip邮编下的”2860”出现了两次。如果我们想把相同zip下两个DataFrame的数据信息合并，我们就得用到横向合并，concat()提供了一个非常方便的办法就是concat(axis=1)或者concat(axis=’columns’)就可以实现横向合并了：

```python
In [7]: pd.concat([population, unemployment], axis=1)
Out[17]:
|       | 2010 Census Population participants unemployment | participants | unemployment |
|-------|--------------------------------------------------|--------------|--------------|
| 1097  | NaN                                              | 0.33         | 42.0         |
| 2860  | 45199.0                                          | 0.11         | 34447.0      |
| 37660 | 40038.0                                          | NaN          | NaN          |
| 46167 | NaN                                              | 0.02         | 4800.0       |
| 57538 | 322.0                                            | NaN          | NaN          |
| 59916 | 130.0                                            | NaN          | NaN          |
| 80808 | NaN                                              | 0.07         | 4310.0       |
```

### concat() Vs. merge()

在上面说完了`concat()`和`append()`横向纵向的连接与合并之后，下面要说一下`concat()`和`merge()`的区别和关系。上面我们说了`concat()`和`merge()`都可以进行横纵向的合并，在用法上和输出结果上两者有一些区别。这里要引入join的概念。`concat()`的默认join方式是outer join，而`merge()`的默认join方式是inner join。另外`concat()`和`merge()`在合并DataFrame的时候还有一个重要的区别就是，`concat()`是通过index来合并的，而`merge()`是通过列明（column label ）来合并的，如果列名设置成为了index的话需要把用来合并列名的index去掉之后再进行合并，否则会出现KeyError错误提示找不到列名。下面继续使用population和unemployment两个DataFrame来进行相关展示：

```python
In [1]: population = pd.read_csv('population_00.csv', index_col=0)
In [1]: unemployment = pd.read_csv('unemployment_00.csv', index_col=0)
In [1]: print(population)
|               | 2010 Census Population |
|---------------|------------------------|
| Zip Code ZCTA |                        |
| 57538         | 322                    |
| 59916         | 130                    |
| 37660         | 40038                  |
| 2860          | 45199                  |
In [2]: print(unemployment)
|       | unemployment | participants |
|-------|--------------|--------------|
| Zip   |              |              |
| 2860  | 0.11         | 34447        |
| 46167 | 0.02         | 4800         |
| 1097  | 0.33         | 42           |
| 80808 | 0.07         | 4310         |

In [3]: pd.concat([population, unemployment], axis=1) #pd.concat(join='outer')默认值为outer
Out[3]:
|       | 2010 Census Population participants unemployment | participants | unemployment |
|-------|--------------------------------------------------|--------------|--------------|
| 1097  | NaN                                              | 0.33         | 42.0         |
| 2860  | 45199.0                                          | 0.11         | 34447.0      |
| 37660 | 40038.0                                          | NaN          | NaN          |
| 46167 | NaN                                              | 0.02         | 4800.0       |
| 57538 | 322.0                                            | NaN          | NaN          |
| 59916 | 130.0                                            | NaN          | NaN          |
| 80808 | NaN                                              | 0.07         | 4310.0       |

In [4]: pd.concat([population, unemployment], axis=1, join='inner') #pd.concat(join='outer')默认值为outer，这里把join设置成了inner
Out[4]:
|       | 2010 Census Population participants unemployment | participants | unemployment |
|-------|--------------------------------------------------|--------------|--------------|
| 2860  | 45199.0                                          | 0.11         | 34447.0      |
```

接下来是对相同df进行merge操作：

```python
In [5]: population = pd.read_csv('population_00.csv', index_col=0)
In [5]: unemployment = pd.read_csv('unemployment_00.csv', index_col=0)
    
#这里的导入我们还是设置了第一列ZipCode和Zip为各df的index，然后看一下使用merge()的时候会出现什么情况

In [5]: pd.merge(population, unemployment, left_on='ZipCode', right_on='Zip')
Out[5]: KeyError: "None of ['ZipCode'] are in the columns"
        
#因为ZipCode被设置成了index所以merge找不到该列名，无法进行merge，我们可以.reset_index()，或者在导入数据的时候不设置index就可以解决该问题。

In [6]: population = population.reset_index()
In [6]: unemployment = unemployment.reset_index()
In [6]: pd.merge(population, unemployment, left_on='ZipCode', right_on='Zip')
    
#pd.merge(how='inner')默认值为inner，merge()的合并方式参数是how不是join

Out[6]:
|   | ZipCode | 2010 Census Population | Zip  | Unemployment | Participants |
|---|---------|------------------------|------|--------------|--------------|
| 0 | 2860    | 45199                  | 2860 | 0.11         | 34447        |

#merge的join和concat的join出来的结果会有一些不同，concat出来的df没有index，merge出来的df会有默认index和两个df合并的的列ZipCode和Zip

In [7]: pd.merge(population, unemployment, left_on='ZipCode', right_on='Zip',
               how='outer')
Out[7]: 
|   | ZipCode | 2010 Census Population | Zip     | Unemployment | Participants |
|---|---------|------------------------|---------|--------------|--------------|
| 0 | 57538.0 | 322.0                  | NaN     | NaN          | NaN          |
| 1 | 59916.0 | 130.0                  | NaN     | NaN          | NaN          |
| 2 | 37660.0 | 40038.0                | NaN     | NaN          | NaN          |
| 3 | 2860.0  | 45199.0                | 2860.0  | 0.11         | 34447.0      |
| 4 | NaN     | NaN                    | 46167.0 | 0.02         | 4800.0       |
| 5 | NaN     | NaN                    | 1097.0  | 0.33         | 42.0         |
| 6 | NaN     | NaN                    | 80808.0 | 0.07         | 4310.0       |
#这里有点奇怪，ZipCode和Zip经过outer join之后变成了float类型。
```

我暂且认为更改ZipCode和Zip的这个行为是个bug，并且已经提交给git了。可以看下之后的反馈：https://github.com/pandas-dev/pandas/issues/34017

当然还是有一些办法去解决这个问题，可是使用会concat()方法来进行合并，或者我们可以通过统一两个DataFrame邮编的label来使用on= [‘Zip’]来进行合并，实验表明通过on= [‘Zip’]进行合并不会出现上述问题：

```python
In [8]: population.rename(columns={'ZipCode':'Zip'}, inplace=True) #更改population中的column label
In [8]: merge_2 = pd.merge(population, unemployment, on=['Zip'], how='outer')
print(merge_2)
Out[8]:
|   | Zip   | 2010 Census Population | Unemployment | Participants | Participants |
|---|-------|------------------------|--------------|--------------|--------------|
| 0 | 57538 | 322.0                  | NaN          | NaN          | NaN          |
| 1 | 59916 | 130.0                  | NaN          | NaN          | NaN          |
| 2 | 37660 | 40038.0                | NaN          | NaN          | NaN          |
| 3 | 2860  | 45199.0                | 0.11         | 34447.0      | 34447.0      |
| 4 | 46167 | NaN                    | 0.02         | 4800.0       | 4800.0       |
| 5 | 1097  | NaN                    | 0.33         | 42.0         | 42.0         |
| 6 | 80808 | NaN                    | 0.07         | 4310.0       | 4310.0       |
```

### join() Vs. concat()

join有四种合并方法，分别是`how='left`‘, `how='right'`, `how='inner'`和`how='outer'`。当然这些合并方法`merge()`也是全部都有的。所以看到这里也应该对`append()`, `concat()`, `join()`和`merge()`有很充分的理解了。`merge()`是四个函数里面最强大的，但是在使用原则上来讲并不是每次对数据操作都要用`merge()`，有时候`append()`和`concat()`使用起来可能会更加方便，在最后会总结一下这四个函数的分类和原则。这里先看一下`join()`的实际操作：

```python
In [1]: population.join(unemployment) #join的默认合并方式是how='left'
Out[1]:
|         | 2010 Census Population | unemployment | participants |
|---------|------------------------|--------------|--------------|
| ZipCode |                        |              |              |
| 57538   | 322                    | NaN          | NaN          |
| 59916   | 130                    | NaN          | NaN          |
| 37660   | 40038                  | NaN          | NaN          |
| 2860    | 45199                  | 0.11         | 34447.0      |
```

df1.join(df2, how=’left’)的意思是指以左边的DataFrame为准进行合并，population在unemployment左边，所以这个合并就会以population的index也就是ZipCode为准进行合并。所以df1.join(df2, how=’right’)就会以unemployment的index进行合并：

```python
In [2]: population.join(unemployment, how= 'right')
Out[2]:
|       | 2010 Census Population | unemployment | participants |
|-------|------------------------|--------------|--------------|
| Zip   |                        |              |              |
| 2860  | 45199.0                | 0.11         | 34447        |
| 46167 | NaN                    | 0.02         | 4800         |
| 1097  | NaN                    | 0.33         | 42           |
| 80808 | NaN                    | 0.07         | 4310         |
```

join和concat都是要以index来进行合并，所以在合并时，必须要有对应的index。concat相比join缺少了left和right两种合并方式，但是在outer和inner合并方式来讲得到的结果是一模一样的：

```python
population.join(unemployment, how='outer')
pd.concat([population, unemployment], join='outer', axis=1)
#以上两者结果相同
population.join(unemployment, how='inner')
pd.concat([population, unemployment], join='inner', axis=1)
#以上两者结果相同
```

### append(), concat(), join()和merge()总结

#### append()

语法：`df1.append(df2)`

说明：`append()`就是简单的把两个DataFrame纵向罗列起来，**不需要index**。

#### concat()

语法：`pd.concat([df1, df2])`

说明：`concat()`可以横纵向合并多行或者多列，可以使用inner或者outer方式来合并，**需要index**。

#### join()

语法：`df1.join(df2)`

说明：`join()`可以使用多种合并方式，除了inner和outer之外还可以用left和right，这些操作同样**需要index**。

#### merge()

语法：`pd.merge([df1, df2])`

说明：方式最多的合并函数。**不需要index。**

### merge_order()函数

`merge_order()`函数可以用一个函数进行两个操作，即`merge()`和`sort_value()`。

```python
pd.merge_ordered(hardware, software, on=['', ''], suffixes=['', '']，fill_method='ffill')
```

来自于https://mingju.net/2020/05/merging-dataframes-with-pandas/