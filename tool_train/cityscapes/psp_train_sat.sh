#!/bin/sh
PYTHON=/mnt/proj58/xgxu/anaconda3/bin/python

dataset=cityscapes
exp_name=pspnet50_sat
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool_train/cityscapes/psp_train_sat.sh tool_train/train_sat_psp.py ${config} ${exp_dir}

export PYTHONPATH=./
#srun -p $PARTITION -n1 --gres=gpu:8 --ntasks-per-node=8 --job-name=python \
$PYTHON -u tool_train/train_sat_psp.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.log
