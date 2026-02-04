---
title: TDD
tags: 'Software Engineering'
categories: Study Notes
password: yuanlainiyewanyuanshen
abbrlink: 715050f6
date: 2022-07-18 16:22:18
mathjax:
copyright:
lang: en
---

# TDD Demo

# Tasking

## Tasking Theory

### Tasking Iron Triangle

- Valuable
  - Has business value
  - Implements a feature
  - Users can use it and feel the software change

- Small Enough
  - Let workers start working
  - Not "can't start" or "working blindly"
  - Relatively small for individuals, not too small to code level

- Speak Human Language
  - Communication: understandable after 3 days
  - Product thinking - callback to value

# Clean Code

## About Writing Code

> Any fool can write code that machines can read. Good programmers write code that humans can read.

## 1-10-50 Rule

- Max 1 level of indentation per method
- Max 10 lines per method
- Max 50 lines per class

## SOLID Principles

- Single Responsibility
- Open/Closed
- Liskov Substitution
- Interface Segregation
- Dependency Inversion

# TDD

## Three Laws of TDD

1. Write failing test first
2. Write minimal code to pass
3. Refactor

## Test Structure: Given-When-Then (AAA)

- Arrange (Given)
- Act (When)
- Assert (Then)

## Integration Test vs Unit Test

Integration Test: Tests with real files/databases
Unit Test: Tests a single section only
