#!/bin/bash
git config --global user.email 'dean0731@qq.com'
git config --global user.name '华为主机'

git fetch --all && git reset --hard origin/master && git pull
git fetch --all && git reset --hard origin/dev && git pull

git rm -r --cached .
git add .
git commit -m 'update .gitignore'

git remote 查看所有远程仓库
git remote -v 查看指定远程仓库地址
git remote add  origin https://github.com/Dean0731/Segmentation.git  添加远程仓库
git remote set-url origin 你新的远程仓库地址remote  更新远程仓库地址

git branch 当前分支
git branch -a 所有当前分支
git add . && git commit -m 'update' && git push

# 同步分支情况
git remote prune origin

