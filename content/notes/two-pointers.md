---
title: two pointers
tags:
  - Algorithm
  - LeetCode
categories: 刷题找工作
abbrlink: 8f8ed91a
date: 2022-03-11 14:54:41
mathjax:
copyright:
---

双指针

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

这题如果用双指针怎么写呢。

归并排序？

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

因为是按顺序排列的，比较第一个数的绝对值和最后一个输的绝对值，然后把大的放到后面就行。

记得要新创建一个vector

