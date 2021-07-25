for i in 10 20 30 40 50 60 70 80 90 100
do
	echo $i
	./tools/dist_test.sh configs/fvnet/base.py work_dirs/lr_0.003_wd_0.0001/epoch_$i.pth 4 --eval bbox
done