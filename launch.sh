#!/bin/bash

# rates=("0.0002" "5e-05" "8e-06")
# scheduler=("cosine" "constant")
# warmup=("0.05" "0.3")

rates="8e-06"
scheduler="cosine"
warmup="0.3"

# for rate in "${rates[@]}"
# do
#     for sch in "${scheduler[@]}"
#     do
#         for w in "${warmup[@]}"
#         do
            accelerate launch seq2seq.py \
            --mixed_precision bf16 \
            --use_lora \
            --constrained_generation \
            --train_tsvs dataset/italian/train.tsv \
            --dev_tsvs dataset/italian/val.tsv \
            --test_tsvs dataset/italian/test_new.tsv \
            --num_beams 3 \
            --num_return_sequences 3 \
            --model_name_or_path Pretrain-FBK-NLP/mt5-large_ClinicalWhole_${rates}_${scheduler}_${warmup}_512_chain \
            --per_device_train_batch_size 16 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 16 \
            --learning_rate 2e-4 \
            --optim adamw8bits \
            --lr_scheduler_type constant \
            --num_warmup_steps 24 \
            --num_train_epochs 10 \
            --eval_every_epochs 2 \
            --max_source_length 256 \
            --max_target_length 256 \
            --output_dir results/ner-llama-italian-${rates}-${scheduler}-${warmup} \
            --project_name ner-llama \
            --add_labels_as_tokens
#         done
#     done
# done