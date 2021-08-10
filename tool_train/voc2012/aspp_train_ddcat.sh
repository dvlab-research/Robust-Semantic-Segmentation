#!/bin/sh
PYTHON=/mnt/proj58/xgxu/anaconda3/bin/python

dataset=voc2012
exp_name=aspp_ddcat
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool_train/voc2012/aspp_train_ddcat.sh tool_train/train_ddcat_aspp.py ${config} ${exp_dir}

export PYTHONPATH=./
#srun -p $PARTITION -n1 --gres=gpu:8 --ntasks-per-node=8 --job-name=python \
$PYTHON -u tool_train/train_ddcat_aspp.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.log
