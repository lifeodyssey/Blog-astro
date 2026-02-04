---
title: lfs
tags:
  - python
  - git
categories: Learning Notes
mathjax: true
abbrlink: 442707f1
date: 2021-11-29 12:06:20
copyright:
password:
lang: en
---

GitHub Large Files

<!-- more -->

Recently I've been frequently encountering this error:

fatal: sha1 file '<stdout>' write error: Broken pipe

The reason is that GitHub only supports files up to 100MB. This is usually enough for regular code, but if you have some HDF or NC files, it won't work.

After installing LFS from https://git-lfs.github.com/, you can use the command line (PyCharm doesn't seem to work).

Actually, the best approach is to store data files separately, but I once committed the files along with the code, and even if I change the path later, they will still be pushed.

So I had to upload them first, then move the path, then commit and push.

Also, LFS has limitations too - you can only upload 1GB, and the rest requires payment. It's really quite troublesome.

I haven't found a better solution yet.

## [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)

A better solution came just in time.

I found it on [this page](https://stackoverflow.com/questions/2100907/how-to-remove-delete-a-large-file-from-commit-history-in-the-git-repository).

If you've already uploaded some files to GitHub, you can download BFG.

Then:

```bash
$ git clone --mirror git://example.com/some-big-repo.git
$ java -jar bfg.jar --strip-blobs-bigger-than 100M my-repo.git
$ git gc --prune=now --aggressive
```

Done.

Note that this method requires JRE.

## git-filter-branch

But the above method is a dead loop for me.

Because the file is too large to push to GitHub.

So there's no way to clone that mirror and clean the commit.

Another method is:

```bash
$ git filter-branch -f --index-filter "git rm -rf --cached --ignore-unmatch *.HDF" -- --all
```

For me, I know the large files are all HDF, so I can just use wildcards.

Then:

```bash
$ rm -rf .git/refs/original/
$ git reflog expire --expire=now --all
$ git gc --prune=now
$ git gc --aggressive --prune=now
```

Finally, push them all:

```bash
$ git push --all --force
```
