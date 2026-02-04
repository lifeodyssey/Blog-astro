---
title: PyTorch基礎
tags:
  - Machine Learning
  - Inversion
  - Deep Learning
  - PyTorch
categories:
  - 学習ノート
abbrlink: fee729e7
slug: pytorch-basics
date: 2021-12-31 10:43:02
mathjax: true
copyright: true
lang: ja
---

まさか自分がニューラルネットワークの訓練を始める日が来るとは思わなかった。

<!-- more -->

主な参考資料は[こちら](https://github.com/zergtant/pytorch-handbook)

# PyTorch基礎

## テンソル

numpyの基本単位はndarrayで、PyTorchの基本単位はtensorです。tensorはGPUで計算できます。

基本操作は同じで、ones_like、zeros、sizeなど、インデックス方法も同じです。

基本的なデータ型は5種類あります：

- 32ビット浮動小数点：torch.FloatTensor（デフォルト）
- 64ビット整数：torch.LongTensor
- 32ビット整数：torch.IntTensor
- 16ビット整数：torch.ShortTensor
- 64ビット浮動小数点：torch.DoubleTensor

これらの数値型以外に、byteとchar型もあります。

### テンソルの特殊操作

加算方法1：

```python
y = torch.rand(5, 3)
print(x + y)
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

加算方法2：

```python
print(torch.add(x, y))
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

出力テンソルを引数として提供：

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

インプレース置換：

```python
# xをyに加算
y.add_(x)
print(y)
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

#### 注意

``_``で終わる操作は、結果で元の変数を置き換えます。例：``x.copy_(y)``、``x.t_()``はすべて``x``を変更します。

次元の変更：

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # size -1は他の次元から推論
print(x.size(), y.size(), z.size())
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

要素が1つだけのテンソルの場合、`.item()`を使用してPythonの数値を取得：

```python
x = torch.randn(1)
print(x)
print(x.item())
tensor([-0.2368])
-0.23680149018764496
```

### NumPyとの変換

Torch TensorとNumPy配列は基礎となるメモリアドレスを共有しており、一方を変更すると他方も変化します。

Torch TensorをNumPy配列に変換：

```python
a = torch.ones(5)
print(a)
tensor([1., 1., 1., 1., 1.])
```

```python
b = a.numpy()
print(b)
[1. 1. 1. 1. 1.]
```

numpy配列の値がどのように変化するか観察：

```python
a.add_(1)
print(a)
print(b)
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

NumPy配列をTorch Tensorに変換：

from_numpyを使用して自動変換：

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

すべてのTensor型はデフォルトでCPUベースです。CharTensorはNumPyへの変換をサポートしていません。

`.to`メソッドを使用してTensorを任意のデバイスに移動：

```python
# is_available関数でcudaが使用可能か確認
# ``torch.device``でテンソルを指定デバイスに移動
if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDAデバイスオブジェクト
    y = torch.ones_like(x, device=device)  # GPUで直接テンソルを作成
    x = x.to(device)                       # または``.to("cuda")``でcudaに移動
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to``は変数の型も変更可能
tensor([0.7632], device='cuda:0')
tensor([0.7632], dtype=torch.float64)
```

## 自動微分

ここが何に使われるのかよく分かりませんでした。

## ニューラルネットワークパッケージ＆オプティマイザ

torch.nnはニューラルネットワーク専用に設計されたモジュラーインターフェースです。nnはAutogradの上に構築され、ニューラルネットワークの定義と実行に使用できます。ここでは主にいくつかの一般的なクラスを紹介します。

**規約：便宜上、torch.nnのエイリアスをnnに設定します。この章ではnn以外にも他の命名規約があります。**

```python
# まず関連パッケージをインポート
import torch
# torch.nnをインポートしてエイリアスを設定
import torch.nn as nn
# バージョンを表示
torch.__version__
```

```bash
'1.0.0'
```

nnエイリアス以外に、nn.functionalもインポートします。このパッケージにはニューラルネットワークで使用される一般的な関数が含まれています。これらの関数の特徴は、学習可能なパラメータを持たないことです（ReLU、pool、DropOutなど）。これらの関数はコンストラクタに配置してもしなくても良いですが、配置しないことを推奨します。

一般的に、**nn.functionalは大文字のFに設定**して呼び出しやすくします：

```python
import torch.nn.functional as F
```

### ネットワークの定義

PyTorchには既製のネットワークモデルが用意されています。nn.Moduleを継承してforwardメソッドを実装するだけで、PyTorchはautogradに基づいてbackward関数を自動的に実装します。forward関数では、tensorがサポートする任意の関数を使用でき、if、forループ、print、logなどのPython構文も使用できます。

```python
class Net(nn.Module):
    def __init__(self):
        # nn.Moduleサブクラスはコンストラクタで親クラスのコンストラクタを実行する必要がある
        super(Net, self).__init__()

        # 畳み込み層 '1'は入力画像が単一チャンネル、'6'は出力チャンネル数、'3'は3*3カーネル
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 線形層、入力1350特徴、出力10特徴
        self.fc1   = nn.Linear(1350, 10)  # 1350はどう計算？下のforward関数を参照
    # 順伝播
    def forward(self, x):
        print(x.size()) # 結果：[1, 1, 32, 32]
        # 畳み込み -> 活性化 -> プーリング
        x = self.conv1(x) # 畳み込みサイズ公式により、結果は30
        x = F.relu(x)
        print(x.size()) # 結果：[1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2)) # プーリング層使用、結果は15
        x = F.relu(x)
        print(x.size()) # 結果：[1, 6, 15, 15]
        # reshape、'-1'は適応的
        # [1, 6, 15, 15]を[1, 1350]に平坦化
        x = x.view(x.size()[0], -1)
        print(x.size()) # これがfc1層の入力1350
        x = self.fc1(x)
        return x

net = Net()
print(net)
```

```bash
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=1350, out_features=10, bias=True)
)
```

ネットワークの学習可能なパラメータはnet.parameters()で返されます：

```python
for parameters in net.parameters():
    print(parameters)
```

```bash
Parameter containing:
tensor([[[[ 0.2745,  0.2594,  0.0171],
          [ 0.0429,  0.3013, -0.0208],
          [ 0.1459, -0.3223,  0.1797]]],
        ...
        [[[ 0.1691, -0.0790,  0.2617],
          [ 0.1956,  0.1477,  0.0877],
          [ 0.0538, -0.3091,  0.2030]]]], requires_grad=True)
...
```

net.named_parametersは学習可能なパラメータとその名前を同時に返します：

```python
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())
```

```bash
conv1.weight : torch.Size([6, 1, 3, 3])
conv1.bias : torch.Size([6])
fc1.weight : torch.Size([10, 1350])
fc1.bias : torch.Size([10])
```

```python
input = torch.randn(1, 1, 32, 32) # ここの入力はforwardの32に対応
out = net(input)
out.size()
```

```bash
torch.Size([1, 10])
```

逆伝播の前に、すべてのパラメータの勾配をゼロにする必要があります：

```python
net.zero_grad()
out.backward(torch.ones(1,10)) # 逆伝播はPyTorchが自動実装
```

**注意**：torch.nnはミニバッチのみをサポートし、一度に1サンプルの入力はサポートしません。つまり、必ずバッチである必要があります。

つまり、1サンプルを入力しても、バッチ処理されます。そのため、すべての入力に次元が追加されます。上記の入力と比較すると、nnは3次元として定義していますが、手動で1次元追加して4次元にしました。最初の1がbatch-sizeです。

ここでコードを見るよりも、courseraを復習する必要があると感じました。

#### 順伝播、逆伝播、ニューラルネットワーク

[これ](https://www.techbrood.com/zh/news/ai/%E7%94%A8%E4%BA%BA%E8%AF%9D%E8%A7%A3%E9%87%8A%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E9%87%8C%E9%9D%A2%E7%9A%84%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%92%8C%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.html)を見つけました

![1.png](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/202112311621806.png)

これは隠れ層のないニューラルネットワークです。左が入力、右が出力。順伝播は入力から出力へ、逆伝播は後ろで得たLossを前の出力層（logisticなど）に渡すことです。

![1552807616619421](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image/202112311632948.png)

これは隠れ層のあるニューラルネットワークです。以前の隠れ層がない場合は、1つのパラメータを調整するだけでした。今は小さな女の子のパラメータも調整する必要があるので、連鎖律で勾配降下を使用します。

これらの中間層と出力は伝説の[活性化関数](https://ja.wikipedia.org/wiki/%E6%B4%BB%E6%80%A7%E5%8C%96%E9%96%A2%E6%95%B0)です。活性化関数がないと、各層の出力は前の層の入力の線形関数となり、非線形関数を適合できません。

入力データに重み（weight）を掛けてバイアス（bias）を加え、活性化関数を適用してそのニューロンの出力を得て、この出力を次の層のニューロンに渡します。これらのweightとbias、さらにL1、L2、batch sizeなどがハイパーパラメータです。

### 損失関数

## 損失関数

nnでは、PyTorchは一般的な損失関数も提供しています。以下ではMSELossを使用して平均二乗誤差を計算します：

```python
y = torch.arange(0,10).view(1,10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)
# lossはスカラーなので、itemで直接Pythonの数値を取得可能
print(loss.item())
```

28.92203712463379

## オプティマイザ

## オプティマイザ

逆伝播ですべてのパラメータの勾配を計算した後、最適化手法を使用してネットワークの重みとパラメータを更新する必要があります。例えば、確率的勾配降下法（SGD）の更新戦略は：

weight = weight - learning_rate * gradient

torch.optimにはRMSProp、Adam、SGDなど、ほとんどの最適化手法が実装されています。以下ではSGDを使用した簡単な例を示します：

```python
import torch.optim
```

```python
out = net(input) # ここで呼び出すとforward関数のxのサイズが表示される
criterion = nn.MSELoss()
loss = criterion(out, y)
# 新しいオプティマイザを作成、SGDは調整するパラメータと学習率のみ必要
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
# まず勾配をゼロに（net.zero_grad()と同じ効果）
optimizer.zero_grad()
loss.backward()

# パラメータを更新
optimizer.step()
```

```bash
torch.Size([1, 1, 32, 32])
torch.Size([1, 6, 30, 30])
torch.Size([1, 6, 15, 15])
torch.Size([1, 1350])
```

これでPyTorchを使用したニューラルネットワークの完全なデータ伝播が完了しました。

後は詳しく見ていません。

## ファインチューニング

これはちょうど自分のニーズに合っています。

あるタスクに対して、訓練データが少ない場合はどうすればいいでしょうか？心配いりません。まず同様の他人が訓練したモデルを見つけ、その既製の訓練済みモデルを取り、自分のデータに置き換え、パラメータを調整して再訓練します。これがファインチューニングです。PyTorchで提供されている古典的なネットワークモデルは、すべて公式がImagenetデータセットで事前訓練したものです。訓練データが不足している場合、これらをベースモデルとして使用できます。

1. 新しいデータセットが元のデータセットと類似している場合、最後のFC層を直接ファインチューニングするか、新しい分類器を指定できます
2. 新しいデータセットが小さく、元のデータセットとかなり異なる場合、モデルの中間から訓練を開始し、最後の数層のみをファインチューニングできます
3. 新しいデータセットが小さく、元のデータセットとかなり異なり、上記の方法でもうまくいかない場合、再訓練が最善で、事前訓練モデルは新しいモデルの初期化データとしてのみ使用します
4. 新しいデータセットのサイズは元のデータセットと同じである必要があります。例えば、CNNでは入力画像サイズが同じでないとエラーになります
5. データセットサイズが異なる場合、最後のfc層の前に畳み込みまたはプール層を追加して最終出力をfc層と一致させることができますが、精度が大幅に低下するため推奨しません
6. 異なる層に異なる学習率を設定できます。一般的に、元のデータで初期化された層の学習率は初期化学習率より小さく（通常10倍小さく）設定することを推奨します。これにより、既に初期化されたデータが急速に歪むことを防ぎ、初期化学習率を使用する新しい層は素早く収束できます。

詳細は[こちら](https://github.com/zergtant/pytorch-handbook/blob/master/chapter4/4.1-fine-tuning.ipynb)を参照

[可視化](https://github.com/zergtant/pytorch-handbook/blob/master/chapter4/4.2.2-tensorboardx.ipynb)の内容もあり、必要になったら追加します。

