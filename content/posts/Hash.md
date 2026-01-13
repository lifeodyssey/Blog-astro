---
title: Hash
tags:
  - Algorithm
  - LeetCode
categories: 刷题找工作
abbrlink: 718acd86
date: 2022-03-10 20:32:35
mathjax:
copyright:
---

这次这个来自于

<!-- more -->

# 经典问题引入

给出N个正整数，再给出M个正整数，问这M个数中的每个数是否在N中出现过。

一个最直观暴力的方法是，对于M里的每个数都去N里查一遍，这样时间复杂度是o(mn)，当MN都很大的时候，这个时间是无法承受的。

所以另一个办法就是拿空间换时间，即设定一个bool array hashTable[N], if x appear in N, hashTable[x]=true, else hashTable[x]=false, thus the code is 

```c
#include<cstdio>
const int maxn=1000010;
bool hashTable[maxn]={false};//init as false
int main()
{
    int n,m,x;
    scanf("%d%d",&n,&m);
    for (int i=0;i<n;i++)
    {
        scanf("%d",&x);
        hashTable[x]=true;//appeared
    }
        for (int i=0;i<m;i++)
    {
        scanf("%d",&x);
        if(hashTable[x]==true){
         printf("Yes");   
        }//appeared
        else
        {
        printf("No")    
        }
    }
    
}
```

Onething I do not understand is that where is the input of x?

Maybe need to review again when I meet similar problem

