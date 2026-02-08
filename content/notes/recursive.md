---
title: recursive
tags:
  - Algorithm
  - LeetCode
categories: 刷题找工作
abbrlink: b8c59cfd
date: 2022-03-11 10:03:29
mathjax:
copyright:
---

https://www.youtube.com/watch?v=AqGagBmFXgw

太难了 等以后做的多了回来总结

<!-- more -->

# Recursion intro

Recursion is an approach to solving problems using a function calls itself as a subroutine

to divide the whole big problem into different small problems.

![image-20220311100950791](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111009870.png)

Leetcode 700

先复习一下结构体...

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;指向左节点的指针
 *     TreeNode *right;指向右节点的指针
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}默认生成的构造函数
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

对结构体不熟悉，才做不出来，看了提示才做出来的题目。

以后这种题可以换python

# 三中常见形式

## Memorization

![image-20220311105216687](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111052742.png)

经典问题就是斐波那契数列

![image-20220311105311156](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111053066.png)

Leetcode 509

挺简单的

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

## Divide and conquer

![image-20220311140416967](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111404030.png)

Leetcode 98

又是一个medium

还是对结构体不熟

自己可能需要专门搞一下binary search tree

代码模板

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

## backtracking

太难了55555

