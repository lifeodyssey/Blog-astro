---
title: Obsidian Daily Record System
tags:
  - Random Thoughts
categories: Practical Tools
abbrlink: f36cc8b7
date: 2022-07-23 23:36:22
mathjax:
copyright:
lang: en
---

This is a follow-up to https://lifeodyssey.github.io/posts/b58cbef5.html
First, let me write about the shortcomings of this current solution:
- It can only do daily records, not Personal Knowledge Management (PKM)
- Since I didn't purchase Obsidian's paid service, it relies on git for synchronization, so it can't sync across all platforms. It can only be used on desktop, and mobile still relies on note apps
- Poor support for task lists, this part is still handled by Microsoft ToDo

In the future, I might combine it with Nextcloud.
Mainly completed based on this: https://diygod.me/obsidian/

<!-- more -->

# Problems to Solve

Compared to the original author, I don't like adding too many things to Obsidian. For exercise and fitness I use Google Fit, for sleep I use Android Sleep Companion, for money I use Mint Accounting.

Here I mainly use Obsidian to accomplish the following things:

1. Daily Completed List, write ten things to praise myself
2. Some small diary entries that might come to mind in daily life. Big diaries are written in OneNote and paper journals, this is just like a notepad
3. Kanban, because I don't want to use Trello

So I simplified the original version a lot.

The main requirements are as follows:

1. Click Calendar every day to automatically generate a diary and arrange it in /Year/Month/Week order
2. Generate a weekly plan with Gantt chart and toggle summary
3. Automatically generate a kanban every month
4. Generate a large yearly plan with Gantt chart, and have a reminder to review at the end of the year

Let's go through them one by one.

# Specific Changes

This can be accomplished using Periodic Note and the function in templater. Since I often use a VPN, I didn't use the IP-based method to get the current location, but used a fixed location instead. Taking getWeather as an example:

```bash
curl wttr.in/"$(curl -s --header "user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36" https://api.ip.sb/geoip | /opt/homebrew/bin/jq -r ".city" | sed 's/ /%20/')"\?format="%l+%c%t"
```

I changed it to:

```bash

curl wttr.in/City:Province:Country?format="%l+%c%t"
```

Everything else remains unchanged.
