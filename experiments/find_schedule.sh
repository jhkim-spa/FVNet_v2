for lr in 0.0003 0.001 0.003 0.01
do
	for wd in 0.0003 0.001 0.003 0.01
	do
		./tools/dist_train.sh configs/fvnet/base.py 4\
			--cfg-options checkpoint_config.interval=2\
						  evaluation.interval=2 optimizer.lr=$lr\
						  optimizer.weight_decay=$wd\
						  runner.max_epochs=50
			--work-dir work_dirs/lr_${lr}_wd_${wd}_50e
	done
done

for lr in 0.0003 0.001 0.003 0.01
do
	for wd in 0.0003 0.001 0.003 0.01
	do
		./tools/dist_train.sh configs/fvnet/base.py 4\
			--cfg-options checkpoint_config.interval=2\
						  evaluation.interval=2 optimizer.lr=$lr\
						  optimizer.weight_decay=$wd\
						  runner.max_epochs=100
			--work-dir work_dirs/lr_${lr}_wd_${wd}_100e
	done
done