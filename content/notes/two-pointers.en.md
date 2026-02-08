---
title: Two Pointers
tags:
  - Algorithm
  - LeetCode
categories: Job Hunting Practice
abbrlink: 8f8ed91a
slug: two-pointers
date: 2022-03-11 14:54:41
mathjax: true
copyright: true
lang: en
---

Two Pointers

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

How would you write this problem using two pointers?

Merge sort?

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

Since it's sorted in order, compare the absolute value of the first number with the absolute value of the last number, then put the larger one at the end.

Remember to create a new vector.
