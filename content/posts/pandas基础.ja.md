---
title: Pandas基礎
tags:
  - Research
  - python
categories: 学習ノート
mathjax: true
abbrlink: 50bcdc38
slug: pandas-basics
date: 2020-09-09 09:13:50
copyright: true
lang: ja
---

正直、pandas DataFrameを使いたくないです。いつも使いにくいと感じます。まだスキルレベルが足りないのかもしれません。しかし、研究室のExcelファイルの読み書きやExcelへの出力で使わざるを得ないことが多いです。以前使っていたseabornもpandasベースなので、基本操作を復習しましょう。

<!-- more -->

# 読み書きと作成

## データの読み書き

### SQL

- **読み込み**

```python
import pandas as pd
from sqlalchemy import create_engine
db = create_engine("mysql+pymysql://username:password@localhost:port(3306)/database_name?charset=utf8")
sql = "select * from text"
df = pd.read_sql(sql, db, index_col="index") # index_colでインデックス列を設定、デフォルトは自動生成
```

- **書き込み**

```python
sql.to_sql(df, name='test', con=db,
                 if_exists="append", # テーブルが存在する場合: append, replace（削除して再作成）, fail（挿入しない）
                 index=False # dfのインデックスをデータベースに挿入しない
                 )
```

### Excel

- **読み込み**

```python
df = pd.read_excel(r'file_path',
                   sheet_name='シートを指定、デフォルトは最初',
                   index=False, # Excelからインデックスを読み込まない、新しいインデックスを自動生成
                   index_col=0, # 列0をインデックスに設定
                   header=0, # 行nを列名に設定、デフォルトは0、Noneも可（0-n列を自動生成）
                   usecols=[0, 2] # 列0、2のみインポート
                   )
```

- **書き込み**

```python
'''
異なるシートに書き込み
'''
# ワークブックを作成
excelWriter = pd.ExcelFile('file_path/test.xlsx')
# ワークブックに書き込み
df.to_excel(
    excelWriter,
    sheet_name='',
    index=False, # dfのインデックスをExcelに書き込まない
    encoding='utf-8',
    columns=['a', 'b'], # 書き込む列を指定
    na_rep=0, # 欠損値処理（0で埋める）
    inf_rep=0, # 無限大処理（0で埋める）
)
# 保存（保存しないと反映されない）
excelWriter.save()
'''
直接書き込み
'''
df.to_excel('file_path/test.xlsx') # パラメータ: index, encoding, columns, na_rep, inf_rep
```

### CSV

- **読み込み**

```python
df = pd.read(
    r'file_path/test.csv',
    sep="", # 区切り文字を指定、デフォルトはカンマ
    nrows=2, # 読み込む行数を指定
    encoding='utf-8',
    engine='python', # パスに日本語が含まれる場合に追加
    usecols=[0, 2], # 列0、2のみインポート
    index_col=0, # 列0をインデックスに設定
    header=0 # 行nを列名に設定、デフォルトは0、Noneも可
)
```

- **書き込み**

```python
df.to_csv(
    r'file_path/test.csv',
    index=False, # インデックス列を書き込まない
    columns=['a', 'b'], # 書き込む列を指定
    sep=',', # 区切り文字を設定（デフォルトはカンマ）
    na_rep=0, # 欠損値を0で埋める
    encoding='utf-8',
)
```

### TXT

- **読み込み**

```python
pd.read_table(r'file_path/test.txt', sep='') # CSVファイルも読み込み可能
```

## 作成

一番困るのは、DataFrameにはnp.zeros、np.onesのような既存のDataFrameに基づいて空のDataFrameを初期化する関数がないことです。

```python
df_empty=pd.Dataframe(columns=['A','B','C','D'])
```

そのため、既存のDataFrameから列名を抽出して新しいものを作成する方法があります。

# インデックス

Pandasのデータインデックスは本の目次のようなもので、欲しい章をすばやく見つけることができます。大量のデータに対して、合理的でビジネス上意味のあるインデックスを作成することは、データ分析にとって非常に重要です。

## インデックスの理解

以下は簡単なDataFrameのインデックスの例です：

![pandas index](https://www.gairuo.com/file/pic/2020/04/pandas_index_01.jpg)

ここで：
- 行インデックスはデータのインデックス、列インデックスはSeriesを指す
- DataFrameのインデックスはSeriesを形成するSeriesのインデックスでもある
- インデックスを構築するとデータがより直感的になる（例：各行は特定の国のデータ）
- インデックスを構築するとデータ処理が容易になる
- インデックスは重複を許可するが、ビジネス上は通常重複させない

行と列のレベルが多いデータでは[マルチレベルインデックス](https://www.gairuo.com/p/pandas-multiIndex)が発生することがあります。

## インデックスの構築

以前、データを読み込んでDataFrameを生成する際にインデックスを指定できることを学びました：

```python
data = 'https://www.gairuo.com/file/data/dataset/team.xlsx'
df = pd.read_excel(data, index_col='name') # インデックスをnameに設定
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

読み込み時にインデックスを指定しなかった場合、`df.set_index()`を使用して指定できます：

```python
df.set_index('month') # monthをインデックスに設定
df.set_index(['month', 'year']) # monthとyearをマルチレベルインデックスに設定
'''
            sale
month year
1     2012    55
4     2014    40
      2013    84
10    2014    31
'''

s = pd.Series([1, 2, 3, 4])
df.set_index(s) # インデックスを指定
df.set_index([s, 'year']) # カスタムインデックスと既存フィールドの両方を指定
df.set_index([s, s**2]) # 計算されたインデックス

# その他のパラメータ
df.set_index('month', drop=False) # 元の列を保持
df.set_index('month', append=True) # 元のインデックスを保持
df.set_index('month', inplace=True) # インデックスを構築してdfを上書き
```

## インデックスのリセット

既存のインデックスをキャンセルしてやり直したい場合は、`df.reset_index()`を使用できます：

```python
df.reset_index() # インデックスをクリア
df.set_index('month').reset_index() # 何もしないのと同じ
# 元のインデックスを削除、month列がなくなる
df.set_index('month').reset_index(drop=True)
df2.reset_index(inplace=True) # 上書きして有効にする
# yearレベルのインデックスをキャンセル
df.set_index(['month', 'year']).reset_index(level=1)
df2.reset_index(level='class') # 上記と同じ、レベルインデックス名を使用
df.reset_index(level='class', col_level=1) # 列インデックス
# 存在しないレベル名に指定した名前を入力
df.reset_index(level='class', col_level=1, col_fill='species')
```

## インデックスの種類

様々なビジネスデータ処理に対応するため、インデックスには様々なデータ型に対応した異なる種類が定義されています：

### 数値インデックス Numeric Index

以下の種類があります：

- RangeIndex: 単調整数範囲の不変インデックス
- Int64Index: int64型、順序付きスライス可能な集合の不変ndarray
- UInt64Index: 符号なし整数ラベル用
- Float64Index: Float64型

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

### カテゴリインデックス CategoricalIndex

カテゴリは限られた数の（通常は固定の）可能な値（カテゴリ）のみを含むことができます。列挙型のようなもので、性別は男女のみですが、データの各行に出現します。テキストとして処理すると効率が悪くなります。基盤はpandas.Categoricalです。

```python
pd.CategoricalIndex(['a', 'b', 'a', 'b'])
# CategoricalIndex(['a', 'b', 'a', 'b'], categories=['a', 'b'], ordered=False, dtype='category')
```

カテゴリについては後で詳しく説明します。非常に大きなデータセットでのみその利点が発揮されます。

### 間隔インデックス IntervalIndex

```python
pd.interval_range(start=0, end=5)
'''
IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
              closed='right',
              dtype='interval[int64]')
'''
```

### マルチインデックス MultiIndex

チュートリアルの後半で詳しく説明します。

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

### 日時インデックス DatetimeIndex

```python
# ある日付から別の日付まで連続
pd.date_range(start='1/1/2018', end='1/08/2018')
# 開始時間と期間を指定
pd.date_range(start='1/1/2018', periods=8)
# 月単位の期間
pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
# ネストされた期間
pd.period_range(start=pd.Period('2017Q1', freq='Q'),
                end=pd.Period('2017Q2', freq='Q'), freq='M')
```

### 時間差 TimedeltaIndex

```python
pd.TimedeltaIndex(data =['06:05:01.000030', '+23:59:59.999999',
                         '22 day 2 min 3us 10ns', '+23:29:59.999999',
                         '+12:19:59.999999'])
# datetimeを使用
pd.TimedeltaIndex(['1 days', '1 days, 00:00:05',
                   np.timedelta64(2, 'D'),
                   datetime.timedelta(days=2, seconds=2)])
```

### 期間インデックス PeriodIndex

```python
t = pd.period_range('2020-5-1 10:00:05', periods=8, freq='S')
pd.PeriodIndex(t,freq='S')
```

## インデックスオブジェクト

Pandasの行と列のインデックスは実際には`Index`オブジェクトです。`index`オブジェクトを作成する方法は以下の通りです：

### オブジェクトの作成

```python
pd.Index([1, 2, 3])
# Int64Index([1, 2, 3], dtype='int64')
pd.Index(list('abc'))
# Index(['a', 'b', 'c'], dtype='object')
# 名前を定義できる
pd.Index(['e', 'd', 'a', 'b'], name='something')
```

### 表示

```python
df.index
# RangeIndex(start=0, stop=4, step=1)
df.columns
# Index(['month', 'year', 'sale'], dtype='object')
```

### プロパティ

以下のメソッドは`df.columns`にも適用されます（すべてインデックスオブジェクトなので）：

```python
# プロパティ
df.index.name # 名前
df.index.array # 配列
df.index.dtype # データ型
df.index.shape # 形状
df.index.size # 要素数
df.index.values # 配列
# その他、あまり使わない
df.index.empty # 空かどうか
df.index.is_unique # 重複がないか
df.index.names # 名前のリスト
df.index.is_all_dates # すべて日時か
df.index.has_duplicates # 重複があるか
df.index.values # インデックス値の配列
```

### 操作

以下のメソッドは`df.columns`にも適用されます（すべてインデックスオブジェクトなので）：

```python
# メソッド
df.index.astype('int64') # 型変換
df.index.isin() # 存在確認、下記の例を参照
df.index.rename('number') # インデックス名を変更
df.index.nunique() # ユニーク値の数
df.index.sort_values(ascending=False,) # ソート、降順
df.index.map(lambda x:x+'_') # map関数で処理
df.index.str.replace('_', '') # 文字列置換
df.index.str.split('_') # 分割
df.index.to_list() # リストに変換
df.index.to_frame(index=False, name='a') # DataFrameに変換
df.index.to_series() # Seriesに変換
df.index.to_numpy() # numpyに変換
df.index.unique() # 重複削除
df.index.value_counts() # 重複削除とカウント
df.index.where(df.index=='a') # フィルタ
df.index.rename('grade', inplace=False) # インデックス名を変更
df.index.rename(['species', 'year']) # マルチレベル、インデックス名を変更
df.index.max() # 最大値
df.index.argmax() # 最大値のインデックス
df.index.any()
df.index.all()
df.index.T # 転置、マルチレベルインデックスで便利
```

## インデックスの名前変更

行と列のインデックス名を変更します。

```python
# 列インデックスを一対一で変更
df.rename(columns={"A": "a", "B": "c"})
df.rename(str.lower, axis='columns')
# 行インデックスを変更
df.rename(index={0: "x", 1: "y", 2: "z"})
df.rename({1: 2, 2: 4}, axis='index')
# データ型を変更
df.rename(index=str)
# インデックスを再変更
replacements = {l1:l2 for l1, l2 in zip(list1, list2)}
df.rename(replacements)
```

## インデックス名の名前変更

注意：これはインデックスの名前を変更するもので、インデックスや列名自体ではありません：

```python
s.rename_axis("animal") # インデックス名を変更
df.rename_axis(["dow", "hr"]) # マルチレベルインデックス名を変更
df.rename_axis('info', axis="columns") # 行インデックス名を変更
# マルチレベル列インデックス名を変更
df.rename_axis(index={'a': 'A', 'b': 'B'})
# マルチレベル列行インデックス名を変更
df.rename_axis(columns={'name': 's_name', 'b': 'B'})
df.rename_axis(columns=str.upper) # 行インデックス名を大文字に
```

# クエリと変更

## データ検査

データを取得したら、まずスポットチェックが必要です。一方ではデータ構造を理解し、他方ではデータ品質の問題をランダムにチェックします。よく使うメソッド：

| 構文           | 操作                              | 戻り値の型 |
| -------------- | --------------------------------- | ---------- |
| `df.head(n)`   | DataFrameの最初のn行を表示        | DataFrame  |
| `df.tail(n)`   | DataFrameの最後のn行を表示        | DataFrame  |
| `df.sample(n)` | n個のランダムサンプルを表示       | DataFrame  |

上記はすべて行全体を選択します。

### 先頭を表示 df.head()

データを読み込んだ後、通常は先頭データを確認する必要があります：

```python
df.head()
out:
    name team  Q1  Q2  Q3  Q4
0  Liver    E  89  21  24  64
1   Arry    C  36  37  37  57
2    Ack    A  57  60  18  84
3  Eorge    C  93  96  71  78
4    Oah    D  65  49  61  86

# 数量を指定可能
df.head(15)
```

### 末尾を表示 df.tail()

最後の末尾データを表示します。

```python
df.tail()
out:
        name team  Q1  Q2  Q3  Q4
95   Gabriel    C  48  59  87  74
96   Austin7    C  21  31  30  43
97  Lincoln4    C  98  93   1  20
98       Eli    E  11  74  58  91
99       Ben    E  21  43  41  74

# 数量を指定可能
df.tail(15)
```

### サンプルを表示 df.sample()

`df.sample()`はランダムに1つのサンプルデータを返します。

```python
df.sample()
out:
     name team  Q1  Q2  Q3  Q4
79  Tyler    A  75  16  44  63

# 数量を指定可能
df.sample(15)
```

データの切り取り：

```python
# インデックスの前後のデータを削除
df.truncate(before=2, after=4) # インデックス2-4のみ保持
s.truncate(before="60", after="66")
df.truncate(before="A", after="B", axis="columns") # 列を選択
```

## 列操作

以下の両方の方法で列を表すことができます：

```python
df['name'] # 列のSeriesを返す
df.name
df.Q1
# df.1Q 列名が1Qでも使用不可
# df.my name スペースがあると呼び出せない、アンダースコアを追加して処理可能
```

列名が有効なPython変数の場合、属性として直接使用できます。

## 行と列の選択

条件に基づいて一部の列や行を選択する必要がある場合があります。よく使うメソッド：

| 操作                 | 構文            | 戻り値の型 |
| -------------------- | --------------- | ---------- |
| 列を選択             | `df[col]`       | Series     |
| インデックスで行選択 | `df.loc[label]` | Series     |
| 位置で行選択         | `df.iloc[loc]`  | Series     |
| スライスで行選択     | `df[5:10]`      | DataFrame  |
| 式で行フィルタ       | `df[bool_vec]`  | DataFrame  |

## スライス []

リストのようにスライス機能を使用して一部の行データを選択できますが、単一インデックスはサポートされていません：

```python
df[:2] # 最初の2行
df[4:10]
df[:] # すべてのデータ、通常は使わない
df[:10:2] # ステップで取得
s[::-1] # 順序を反転
```

列も選択できます：

```python
df['name'] # 1列のみ、Series
df[['Q1', 'Q2']] # 2列を選択
df[['name']] # 1列を選択、DataFrameを返す、上記との違いに注意
```

## ラベルで .loc

`df.loc()`の形式はdf.loc[<インデックス式>, <列式>]で、式は以下の形式をサポートします：

単一ラベル：

```python
# インデックスを表す、文字列は引用符が必要
df.loc[0] # インデックス0の行を選択
df.loc[8]
```

単一ラベルのリスト：

```python
df.loc[[0,5,10]] # インデックス0、5、10の行
df.loc[['Eli', 'Ben']] # インデックスがnameの場合
# ブール選択、長さはインデックスと一致する必要がある
df.loc[[False, True]*50] # Trueの行を表示、1つおき
```

ラベル付きスライス（開始と終了を含む）：

```python
df.loc[0:5] # インデックススライス、行0-5、5を含む
df.loc['2010':'2014'] # インデックスが時間の場合、文字列でクエリ可能
df.loc[:] # すべて
# このメソッドはSeriesをサポート
```

列フィルタリング、行フィルタリングが必要：

```python
dft.loc[:, ['Q1', 'Q2']] # すべての行、Q1とQ2列
dft.loc[:10, 'Q1':] # 行0-10、Q1以降のすべての列
```

## 位置で .iloc

`df.iloc`は`df.loc`に似ていますが、自然インデックス（行と列の0-nインデックス）のみ使用でき、ラベルは使用できません。

```python
df.iloc[:3]
df.iloc[:]
df.iloc[2:20:3]
s.iloc[:3]
```

## 特定の値を取得 .at

locに似ていますが、単一の特定の値のみを取得します。構造はat[<インデックス>, <列名>]：

```python
# 注：文字列インデックスは引用符が必要
df.at[4, 'Q1'] # 65
df.at['lily', 'Q1'] # 65 インデックスがnameと仮定
df.at[0, 'name'] # 'Liver'
df.loc[0].at['name'] # 'Liver'
# 指定した列の値に対応する他の列の値を取得
df.set_index('name').at['Eorge', 'team'] # 'C'
df.set_index('name').team.at['Eorge'] # 'C'
# 列の指定インデックスの値を取得
df.team.at[3] # 'C'
```

同様に、iatはilocと同じく数値インデックスのみサポート：

```python
df.iat[4, 2] # 65
df.loc[0].iat[1] # 'E'
```

.getは辞書のような操作ができ、値がない場合はデフォルト値を返します（例では0）：

```python
df.get('name', 0) # name列
df.get('nameXXX', 0) # 0、デフォルト値を返す
s.get(3, 0) # 93、Seriesはインデックスを渡して特定の値を返す
df.name.get(99, 0) # 'Ben'
```

## 式フィルタリング

`[]`スライスでは式を使用してフィルタリングできます：

```python
df[df['Q1'] == 8] # Q1が8に等しい
df[~(df['Q1'] == 8)] # 8に等しくない
df[df.name == 'Ben'] # 名前がBen
df.loc[df['Q1'] > 90, 'Q1':] # Q1が90より大きい、Q1のみ表示
df.loc[(df.Q1 > 80) & (df.Q2 < 15)] # and関係
df.loc[(df.Q1 > 90) | (df.Q2 < 90)] # or関係
df[df.Q1 > df.Q2]
```

論理比較関数：

```python
df.eq() # 等しい ==
df.ne() # 等しくない !=
df.le() # 以下 <=
df.lt() # 未満 <
df.ge() # 以上 >=
df.gt() # より大きい >
# すべてaxis{0 or 'index', 1 or 'columns'}をサポート、デフォルトは'columns'
df[df.Q1.ne(89)] # Q1が89に等しくない
df.loc[df.Q1.gt(90) & df.Q2.lt(90)] # and関係 Q1>90 Q2<90
```

その他の関数：

```python
# isin
df[df.team.isin(['A','B'])] # グループAとBを含む
df[df.isin({'team': ['C', 'D'], 'Q1':[36,93]})] # 複雑なクエリ、他の値はNaN
```

## 関数フィルタリング

```python
df[lambda df: df['Q1'] == 8] # Q1が8
df.loc[lambda df: df.Q1 == 8, 'Q1':'Q2'] # Q1が8、Q1 Q2を表示
```

## whereとmask

```python
s.where(s > 90) # 条件に合わない値はNaN
s.where(s > 90, 0) # 条件に合わない値は0
# np.where、80より大きければTrue、そうでなければFalse
np.where(s>80, True, False)
np.where(df.num>=60, '合格', '不合格')

s.mask(s > 90) # 条件に合う値はNaN
s.mask(s > 90, 0) # 条件に合う値は0
```

## query

```python
df.query('Q1 > Q2 > 90') # SQLのwhere文のように直接記述
df.query('Q1 + Q2 > 180')
df.query('Q1 == Q2')
df.query('(Q1<50) & (Q2>40) and (Q3>90)')
df.query('Q1 > Q2 > Q3 > Q4')
df.query('team != "C"')
df.query('team not in ("E","A","B")')
# スペースを含む列名にはバッククォートを使用
df.query('B == `team name`')

# 変数の受け渡しをサポート、例：平均より40点高い
a = df.Q1.mean()
df.query('Q1 > @a+40')
df.query('Q1 > `Q2`+@a')
```

## filter

filterを使用して行名と列名をフィルタリングできます。

```python
df.filter(items=['Q1', 'Q2']) # 2列を選択
df.filter(regex='Q', axis=1) # Qを含む列名
df.filter(regex='e$', axis=1) # eで終わる
df.filter(regex='1$', axis=0) # 正規表現、1を含むインデックス名
df.filter(like='2', axis=0) # インデックスに2を含む
# 2で始まるインデックス、Qを含む列名
df.filter(regex='^2', axis=0).filter(like='Q', axis=1)
```

# 結合と行列の追加

## 列の追加

元のデータが以下のようになっていると仮定：

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'num_legs': [4, 2], 'num_wings': [0, 2]},
                  index=['dog', 'hawk'])
slen = len(df['num_legs'])
```

1）直接代入

```python
df['a'] = pd.Series(np.random.randn(slen), index=df.index) # indexを追加することを忘れずに
df['b'] = None # None値の列を追加
df['c'] = [2, 4] # リストデータを追加
```

2）locメソッド

```python
df.loc[:,'d'] = pd.Series(np.random.randn(slen), index=df.index)
df.loc[:, 'd'] = [2, 4]
```

3）insertメソッド

*insertメソッドの列名は重複不可、更新も不可*

```python
df.insert(len(df.columns), 'e', pd.Series(np.random.randn(slen)), index=df.index)
df.insert(len(df.columns), 'ee', [1,2])
```

4）assignメソッド

assignメソッドのパラメータはSeries、スカラー、リストで、複数列を同時に追加可能

```python
df = df.assign(f=df.num_wings.mean())  # num_wings列の平均値を新しい列fとして使用
df = df.assign(A=df.num_wings.sum(), B=[1,2])  # 列AとBを追加
```

5）concatメソッド

```python
pd.concat([df, pd.Series(['yes', 'yes']).rename('t')], axis=1) # 列tを追加
```

## 行の追加

```python
import pandas as pd
import numpy as np

# 空のDataFrameを作成
df = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])
```

1）locを使用

```python
for i in range(4):
    df.loc[i] = [np.random.randint(-1, 1) for n in range(3)]
    # df.loc[i] = 5 すべての値が5のレコードを追加
```

2）appendを使用

```python
df.append({'lib': 2, 'qty1': 3, 'qty2': 4}, ignore_index=True)

# appendはDataFrameも直接追加可能
df2 = pd.DataFrame([[1,2,3], [2,3,4]], columns=['lib', 'qty1', 'qty2'])
df.append(df2, ignore_index=True)  # ignore_index=Trueでdf2のインデックスを無視
```

## 結合

### append() Vs. concat()

DataFrameを連結または結合する場合、一般的に縦方向と横方向の2つの方法があります。縦方向は2つ以上のDataFrameを縦に（上から下へ）1つのDataFrameに連結します。横方向は同じインデックスがある場合、同じインデックス上のすべての列データを結合します。

DataFrameの連結と結合によく使う関数は`append()`、`concat()`、`merge()`です。append()は縦方向の連結のみ可能で、`concat()`と`merge()`は両方可能です。`concat()`はデフォルトで縦方向連結（`append()`と同じ効果）、横方向結合には`concat(axis=1)`を使用します。

### concat() Vs. merge()

`concat()`と`merge()`の違いについて説明します。両方とも横縦方向の結合が可能ですが、使用法と出力結果にいくつかの違いがあります。

joinの概念を紹介します。`concat()`のデフォルトjoin方式はouter join、`merge()`のデフォルトはinner joinです。もう1つの重要な違いは、`concat()`はインデックスで結合し、`merge()`は列ラベルで結合することです。

```python
# outer joinでconcat（デフォルト）
pd.concat([population, unemployment], axis=1)

# inner joinでconcat
pd.concat([population, unemployment], axis=1, join='inner')
```

merge操作の場合：

```python
# merge()のデフォルトはinner join、パラメータは'how'で'join'ではない
pd.merge(population, unemployment, left_on='ZipCode', right_on='Zip')

# 列がインデックスに設定されている場合、まずreset_index()が必要
population = population.reset_index()
unemployment = unemployment.reset_index()
pd.merge(population, unemployment, left_on='ZipCode', right_on='Zip', how='outer')
```

### join() Vs. concat()

joinには4つの結合方法があります：`how='left'`、`how='right'`、`how='inner'`、`how='outer'`。`merge()`もこれらすべてのメソッドを持っています。

```python
# joinのデフォルト結合方法はhow='left'
population.join(unemployment)

# rightでjoin
population.join(unemployment, how='right')
```

joinとconcatは両方ともインデックスで結合するため、対応するインデックスが必要です。concatはjoinと比較してleftとrightの結合方法がありませんが、outerとinnerの結果は同じです：

```python
population.join(unemployment, how='outer')
pd.concat([population, unemployment], join='outer', axis=1)
# 上記2つの結果は同じ

population.join(unemployment, how='inner')
pd.concat([population, unemployment], join='inner', axis=1)
# 上記2つの結果は同じ
```

### append()、concat()、join()、merge()のまとめ

#### append()

構文：`df1.append(df2)`

説明：`append()`は単純に2つのDataFrameを縦に並べます。**インデックス不要**。

#### concat()

構文：`pd.concat([df1, df2])`

説明：`concat()`は複数の行または列を横縦方向に結合でき、innerまたはouterメソッドを使用できます。**インデックス必要**。

#### join()

構文：`df1.join(df2)`

説明：`join()`は複数の結合方法を使用でき、innerとouter以外にleftとrightも使用できます。これらの操作も**インデックス必要**。

#### merge()

構文：`pd.merge([df1, df2])`

説明：最も多機能な結合関数。**インデックス不要。**

### merge_ordered()関数

`merge_ordered()`関数は1つの関数で2つの操作を実行できます：`merge()`と`sort_values()`。

```python
pd.merge_ordered(hardware, software, on=['', ''], suffixes=['', ''], fill_method='ffill')
```

