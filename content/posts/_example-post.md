---
title: Example Post Title
date: 2024-01-01 12:00:00
tags:
  - Example
  - Tutorial
categories:
  - Getting Started
slug: example-post
abbrlink: ''
mathjax: false
mermaid: false
---

This is an example post demonstrating the expected frontmatter structure.

<!-- more -->

## Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `title` | Yes | Post title |
| `date` | Yes | Publication date (YYYY-MM-DD HH:mm:ss) |
| `tags` | No | List of tags |
| `categories` | No | List of categories |
| `slug` | No | Custom URL slug (priority over abbrlink) |
| `abbrlink` | No | Legacy Hexo short URL (fallback) |
| `mathjax` | No | Enable math rendering |
| `mermaid` | No | Enable diagram rendering |

## URL Routing

Posts are automatically categorized based on tags:

- **Tech posts** (`/tech/posts/`): Posts with technical tags like "python", "Deep Learning", "Algorithm", etc.
- **Life posts** (`/life/posts/`): All other posts

## Multi-language Support

For translated versions, use the naming convention:
- `example-post.md` (default language)
- `example-post.en.md` (English)
- `example-post.ja.md` (Japanese)
