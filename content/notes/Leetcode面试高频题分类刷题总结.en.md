---
title: Leetcode Interview High-Frequency Problems Summary
slug: leetcode-interview-summary
tags:
  - Algorithm
  - LeetCode
categories: Job Hunting
abbrlink: fc1cc4fb
date: 2022-03-10 12:45:24
mathjax:
copyright:
password: LeetJo
lang: en
---

A friend gave me this, so I won't make it public.

<!-- more -->

# Leetcode Interview High-Frequency Problems Summary

Inspired by: https://zhuanlan.zhihu.com/p/349940945

This article contains 200+ problems, with derivatives about 250+. Mostly medium difficulty, few easy, some hard. Enough for most algorithm interviews.

---

**Sort:**
- Basics: Quick Sort & Merge Sort implementation, Time Complexity, In-place, Stability
- Entry: Leetcode 148, 56, 27
- Advanced: Leetcode 179, 75

**Selection:**
- Quick Selection, Heapsort. Know the trade-offs, know the corresponding C++ STL methods. Hand-writing quick selection is sometimes necessary.
  - Leetcode 215, 347, 2099 (clear progression)
  - Leetcode 4. Median of Two Sorted Arrays

**Linked List:**
- Basics: How to implement and traverse a linked list. Head/tail insert/delete are O(1), finding any element is O(N)
- Entry:
  - Leetcode 206. Reverse Linked List
  - Leetcode 876. Middle of the Linked List

Fast/slow pointers and list reversal are fundamental to almost all linked list problems. Reversal code is short - memorize it.

- Advanced:
  - Leetcode 160. Intersection of Two Linked Lists
  - Leetcode 141. Linked List Cycle (and Cycle II)
  - Leetcode 92. Reverse Linked List II
  - Leetcode 328. Odd Even Linked List

**Heap, Dictionary (Set, Map, Hashmap), Stack, Queue:**

- Basics: Read *Introduction to Algorithms* relevant chapters.
- Queue:
  - Leetcode 225, 346, 281, 1429, 54, 362
- Stack:
  - Leetcode 155, 232, 150, 224, 20, 1472, 1209, 1249, 735
- Hashmap/Hashset:
  - Leetcode 1, 146, 128, 73, 380, 49, 350, 299, 348
- Heap/Priority Queue:
  - Leetcode 973, 347, 23, 264, 1086, 88, 692, 378, 295, 767, 1438, 895
- Map:
  - Leetcode 729, 981, 846, 218, 480, 318

**Binary Search:**

- Three templates: [L, R), [L, R], Breakpoint.
- Explicit:
  - Leetcode 34, 33, 1095, 162, 278, 74, 240
- Implicit:
  - Leetcode 69, 540, 644, 528, 1300, 1060, 1062, 1891

**Two Pointer:**

- Basics: Same direction, opposite direction, or facing each other.
- Opposite (mostly palindrome):
  - Leetcode 409, 125, 5
- Facing (two sum based):
  - Leetcode 1, 167, 15, 16, 18, 454, 277, 11
- Same direction (sliding window):
  - Leetcode 283, 26, 395, 340, 424, 76, 3, 1004

**DFS (Depth-First Search):**

- Tree-based DFS:
  - Leetcode 543, 226, 101, 951, 124, 236, 105, 104, 987, 1485, 572, 863, 1110
- BST:
  - Leetcode 230, 98, 270, 235, 669, 700, 108, 333, 285
- Graph-based DFS:
  - Leetcode 341, 394, 51, 291, 126, 93, 22, 586, 301, 37, 212, 1087, 399, 1274, 1376, 694, 131
- Permutation/Combination DFS:
  - Leetcode 17, 39, 78, 46, 77, 698, 526
- Memoization Search:
  - Leetcode 139, 72, 377, 1235, 1335, 1216, 97, 472, 403, 329

**Topological Sort:**
- Leetcode 207, 444, 269, 310, 366

**BFS (Breadth-First Search):**

- Tree-based BFS:
  - Leetcode 102, 103, 297, 314
- Graph-based BFS:
  - Leetcode 200, 133, 127, 490, 323, 130, 752, 815, 1091, 542, 1293

**Prefix Sum:**
- Leetcode 53, 1423, 1031, 523, 304

---

*Above are high-frequency topics. Below are medium-frequency topics.*

**Union Find:**
- Leetcode 721, 547, 737, 305

**Trie (Prefix Tree):**
- Leetcode 208, 211, 1268, 212

**Monotone Stack/Queue:**
- Leetcode 85, 84, 907, 739, 901, 503, 239

**Sweep Line:**
- Leetcode 253, 218, 759

**Dynamic Programming:**
- Leetcode 674, 62, 70, 64, 368, 300, 354, 256, 121, 55, 45
- Leetcode 132, 312, 1143, 1062, 718, 174, 115, 72, 91, 639
- Leetcode 712, 221, 1277, 198, 213, 740, 87, 1140, 322, 518
- Leetcode 1048, 44, 10, 32, 1235, 1043, 926
