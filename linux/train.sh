#!/bin/bash
train='/home/dean/PythonWorkSpace/Segmentation/train.py'
dir=''
if [ -z $1 ]; then
  dir='train.log'
else
  dir=$1
fi
echo $dir
nohup /home/dean/anaconda3/envs/tf2/bin/python $train > $dir 2>&1 &

