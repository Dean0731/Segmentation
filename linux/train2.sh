#!/bin/bash
train='/home/dean/PythonWorkSpace/Segmentation/train2.py'
dir=''
if [ -z $1 ]; then
  dir='train.log'
else
  dir=$1
fi
echo $dir
nohup /home/dean/anaconda3/envs/tf2/bin/python $train > $dir 2>&1 &
#nohup /home/dean/anaconda3/envs/tf2.2/bin/python $train > $dir 2>&1 &
#nohup /home/dean/anaconda3/envs/tf2_cpu/bin/python $train > $dir 2>&1 &

# tf2 gpu 2.3
# tf2.2 gpu 2.2
# tf2_cpu  cpu 2.2

