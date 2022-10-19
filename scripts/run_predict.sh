#!/bin/bash
#SBATCH --mem=30G
#SBATCH --time=5-0
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH -c6

CHECKPOINT='/Users/niko/ML/fusion/final_model/checkpoint-2766'
SEED=1
GPU_NUMBER=0
MODEL_NAME='fusion'
MODEL_NAME_TEXT='nlpaueb/legal-bert-base-uncased'
MODEL_NAME_AMR='/Users/niko/ML/fusion/amrbart/model'
CONCAT_EMB_DIM=1536 # last hidden state size of both models added up
BATCH_SIZE=2
ACCUMULATION_STEPS=4
TASK='logiqa_fusion'
HOME_PATH='/Users/niko/ML/fusion'
DATA_SET_PATH_TEXT='/Users/niko/ML/fusion/data/logiqa/dataset_text'
DATA_SET_PATH_AMR='/Users/niko/ML/fusion/data/logiqa/dataset_amr'

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} HOME_PATH=${HOME_PATH} python experiments/fusion_predict.py \
    --task_name ${TASK} \
    --model_name_or_path_text ${MODEL_NAME_TEXT} \
    --model_name_or_path_amr ${MODEL_NAME_AMR} \
    --checkpoint ${CHECKPOINT} \
    --output_dir logs/${TASK}/${MODEL_NAME}/seed_${SEED} \
    --do_train --do_pred \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --num_train_epochs 1 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --data_set_path_text ${DATA_SET_PATH_TEXT} \
    --data_set_path_amr ${DATA_SET_PATH_AMR} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --concat_emb_dim ${CONCAT_EMB_DIM} \
    --max_seq_length_text 512 \
    --max_seq_length_amr 1024 \
    --max_eval_samples 0 \
    --max_train_samples 1 \
    # max train sample set to 1, therefore Traininer does not actually train, 
    # but prediction is executed.
    #--question needs to be set for logiqa (which has a context and question)

