---
title: 再帰
tags:
  - Algorithm
  - LeetCode
categories: 就活練習
abbrlink: b8c59cfd
slug: recursive
date: 2022-03-11 10:03:29
mathjax: true
copyright: true
lang: ja
---

https://www.youtube.com/watch?v=AqGagBmFXgw

難しすぎる。もっと問題を解いてから戻ってまとめる。

<!-- more -->

# 再帰入門

再帰は、関数が自身をサブルーチンとして呼び出すことで問題を解決するアプローチです。大きな問題全体を異なる小さな問題に分割します。

![image-20220311100950791](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111009870.png)

LeetCode 700

まず構造体を復習...

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left; // 左ノードへのポインタ
 *     TreeNode *right; // 右ノードへのポインタ
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {} // デフォルトコンストラクタ
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
```

```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        //if root is null return null
        if(!root) return root;
        //create node to return
        TreeNode *node=new TreeNode();
        // if root->val != val,search in left and right
        //otherwise this would be required node and we would return it
        if(val<root->val){
            //search in left
            node=searchBST(root->left,val);
        } else if(val>root->val){
            //search in right
            node=searchBST(root->right,val);
        } else {
            //required node
            node=root;
        }
        return node;
    }
};
```

構造体に慣れていなかったので解けなかった。ヒントを見てやっと解けた問題。

今後この種の問題はPythonに切り替えられる。

# 3つの一般的な形式

## メモ化

![image-20220311105216687](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111052742.png)

典型的な問題はフィボナッチ数列です。

![image-20220311105311156](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111053066.png)

LeetCode 509

結構簡単。

```C++
class Solution {
public:
    int fib(int n) {
        int sum=0;

        if (n==2)
        {
            return 1;
        }
        if (n==1)
        {
            return 1;

        }
        if (n==0)
        {
            return 0;
        }
        else
        {
            sum=sum+fib(n-1)+fib(n-2);
            n=n-1;
        }
        return sum;
    }
};
```

## 分割統治法

![image-20220311140416967](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111404030.png)

LeetCode 98

またmedium問題。

まだ構造体に慣れていない。

おそらく二分探索木を専門的に勉強する必要がある。

コードテンプレート：

```python
def divide_and_conquer(S):
    #(1)Divide the problem into a set of subproblem
    [S1,S2,...Sn]=divide(S)
    #2 solve the subproblem
    #obtain the result of subproblem
    ret=[divide_and_conquer(Si) for Si in [S1..Sn]]
    [R1,R2..Rn]=rets
    #3combine
    return combine([R1,R2,..Rn])
```

## バックトラッキング

難しすぎる55555（泣）
