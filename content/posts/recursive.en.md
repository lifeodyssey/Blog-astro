---
title: Recursion
tags:
  - Algorithm
  - LeetCode
categories: Job Hunting Practice
abbrlink: b8c59cfd
slug: recursive
date: 2022-03-11 10:03:29
mathjax: true
copyright: true
lang: en
---

https://www.youtube.com/watch?v=AqGagBmFXgw

Too difficult. Will come back to summarize after doing more problems.

<!-- more -->

# Recursion Intro

Recursion is an approach to solving problems using a function that calls itself as a subroutine to divide the whole big problem into different small problems.

![image-20220311100950791](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111009870.png)

LeetCode 700

First, let me review structs...

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left; // pointer to left node
 *     TreeNode *right; // pointer to right node
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {} // default constructor
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

I wasn't familiar with structs, so I couldn't solve it. Only solved it after looking at hints.

For these kinds of problems, I can switch to Python in the future.

# Three Common Forms

## Memorization

![image-20220311105216687](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111052742.png)

The classic problem is the Fibonacci sequence.

![image-20220311105311156](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111053066.png)

LeetCode 509

Pretty simple.

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

## Divide and Conquer

![image-20220311140416967](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/image.202203111404030.png)

LeetCode 98

Another medium problem.

Still not familiar with structs.

I probably need to specifically study binary search trees.

Code template:

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

## Backtracking

Too difficult 55555 (crying)
