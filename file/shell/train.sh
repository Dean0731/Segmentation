nohup python main.py --type pytorch --epochs 80  --info 1 --log 1 >train.log 2>&1 &
nohup python main.py --type pytorch --epochs 80  --info 0 --log 1 >train.log 2>&1 &


#python main.py --type pytorch --epochs 2 --info 0
#python main.py --type pytorch --epochs 2 --info 0 --log 1
