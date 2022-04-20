#!/bin/bash
#SBATCH --mem=30G
#SBATCH --time=5-0
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH -c6

GPU_NUMBER=0
MODEL_NAME_TEXT='bert-base-uncased'
MODEL_NAME_AMR='/Users/niko/ML/case_hold/amrbart/model'
BATCH_SIZE=4
ACCUMULATION_STEPS=2
TASK='logiqa_amr'
HOME_PATH='/Users/niko/ML/fusion'
DATA_SET_PATH_TEXT='/Users/niko/ML/fusion/data/LogiQADataset/dataset_text'
DATA_SET_PATH_AMR='/Users/niko/ML/fusion/data/LogiQADataset/dataset_amr'

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} HOME_PATH=${HOME_PATH} python experiments/mc.py \
    --task_name ${TASK} \
    --model_name_or_path_text ${MODEL_NAME_TEXT} \
    --model_name_or_path_amr ${MODEL_NAME_AMR} \
    --output_dir logs/${TASK}/${MODEL_NAME}/seed_1 \
    --do_train --do_eval --do_pred \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --num_train_epochs 20 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --seed 1 \
    --data_set_path_text ${DATA_SET_PATH_TEXT} \
    --data_set_path_amr ${DATA_SET_PATH_AMR} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --max_seq_length_text 256 \
    --max_seq_length_amr 1024 \
    --max_train_samples 4 \
    --max_eval_samples 4 \
    --max_predict_samples 4 \
    --overwrite_cache \
    --fp16 \
    --fp16_full_eval \
    --optim adafactor \
    --gradient_checkpointing \
