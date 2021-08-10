#!/bin/sh
PYTHON=/mnt/proj58/xgxu/anaconda3/bin/python


dataset=cityscapes
exp_name=aspp_ddcat
exp_dir=exp/${dataset}/${exp_name}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool_test/cityscapes/aspp_test_ddcat.sh tool_test/cityscapes/test_city_aspp_ddcat.py ${config} ${exp_dir}

export PYTHONPATH=./
#srun -p $PARTITION -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=python \
$PYTHON -u tool_test/cityscapes/test_city_aspp_ddcat.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log


$PYTHON -u tool_test/cityscapes/test_city_aspp_ddcat.py \
  --config=${config} --attack \
  2>&1 | tee ${result_dir}/test-$now.log
