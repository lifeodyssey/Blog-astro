---
title: 双指針法
tags:
  - Algorithm
  - LeetCode
categories: 就活練習
abbrlink: 8f8ed91a
slug: two-pointers
date: 2022-03-11 14:54:41
mathjax: true
copyright: true
lang: ja
---

双指針法（Two Pointers）

<!-- more -->

```c++
//Leetcode977
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        for (int i=0;i<nums.size();i++)
        {
            nums[i]=nums[i]*nums[i];
        }
         sort(nums.begin(),nums.end());
        return nums;
}};
```

この問題を双指針法で書くとどうなるか？

マージソート？

```c++
class Solution {
public:
    vector<int> sortedSquares(vector<int>& A) {
        vector<int> res(A.size());
        int l = 0, r = A.size() - 1;
        for (int k = A.size() - 1; k >= 0; k--) {
            if (abs(A[r]) > abs(A[l])) res[k] = A[r] * A[r--];
            else res[k] = A[l] * A[l++];
        }
        return res;
    }
};
```

![image](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203121401824.png)

順番に並んでいるので、最初の数の絶対値と最後の数の絶対値を比較し、大きい方を後ろに置けばよい。

新しいvectorを作成することを忘れずに。
