
export PYTHONPATH="../":"${PYTHONPATH}"

echo "================== ARTICLE BASELINE ================== "



# Data Version 1 [Given title, article, predict/generate title and article]: DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order_probe 
# Data Version 2 [Given title, article, predicted-title, generate just article]: DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order_probe_v2_pred_titles 


echo "ZEROSHOT BART-CNN/ pegasus-multi_news "
export CUDA=7
export DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order
export BSZ=2
export ACC_STEP=8
export LR=3e-4 # 3e-5
export EPOCH=10 # 20
export CUSTOM_NAME=zeroshot
# export MODEL_NAME=bart-large-cnn
export MODEL_NAME=pegasus-multi_news 
export OUTPUT_DIR=${PWD}/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}
# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=google/$MODEL_NAME \
--learning_rate=${LR} \
--train_batch_size=${BSZ} \
--eval_batch_size=${BSZ} \
--gradient_accumulation_steps ${ACC_STEP} \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=${EPOCH}  \
--gpus=1 \
--do_predict \
--max_source_length 512 \
--max_target_length 250 \
--val_max_target_length 250 \
--eval_max_gen_length 250 \
--test_max_target_length 250 \
--overwrite_output_dir 



echo "SIMPLE FT"
export CUDA=1
export DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order
export BSZ=2
export ACC_STEP=8
export LR=3e-4 # 3e-5
export EPOCH=10 # 20
export CUSTOM_NAME=article_ft
export MODEL_NAME=bart-large
export OUTPUT_DIR=${PWD}/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}.${LR}.512_250.e${EPOCH}.random
# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=facebook/$MODEL_NAME \
--learning_rate=${LR} \
--train_batch_size=${BSZ} \
--eval_batch_size=${BSZ} \
--gradient_accumulation_steps ${ACC_STEP} \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=${EPOCH}  \
--gpus=1 \
--do_train "$@" \
--do_predict \
--max_source_length 512 \
--max_target_length 250 \
--val_max_target_length 250 \
--eval_max_gen_length 250 \
--test_max_target_length 250 \
--overwrite_output_dir 




echo "CLASSIFICATION LOSS"

export CUDA=5
export TASK=lr_vs_roundup #propaganda
export ratio=1
export CUSTOM_NAME=mt_${ratio}_${TASK}

export MODEL_NAME=bart-large
export LR=3e-5
export epoch=20
export OUTPUT_DIR=${PWD}/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}.lr${LR}.bz8
export DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=facebook/$MODEL_NAME \
--learning_rate=${LR} \
--train_batch_size=1 \
--eval_batch_size=1 \
--gradient_accumulation_steps 8 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=${epoch}  \
--gpus=1 \
--do_train "$@" \
--do_predict \
--max_source_length 512 \
--max_target_length 250 \
--val_max_target_length 250 \
--eval_max_gen_length 250 \
--test_max_target_length 250 \
--overwrite_output_dir \
--extra_task $TASK \


echo "======== TRAIN ========== FT with probestyle"
# export CUDA=2
# export LR=3e-5
export CUDA=7
export LR=3e-5
export CUSTOM_NAME=ft_probe_v1
export MODEL_NAME=bart-large
export epoch=20
export OUTPUT_DIR=${PWD}/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}.lr${LR}.bz16
export DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order_probe

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=facebook/$MODEL_NAME \
--eval_batch_size=2 \
--output_dir=$OUTPUT_DIR \
--gpus=1 \
--do_predict \
--max_source_length 512 \
--max_target_length 250 \
--val_max_target_length 250 \
--eval_max_gen_length 250 \
--test_max_target_length 250 \
--overwrite_output_dir 






echo "======== TEST ========== FT with probestyle"
# export CUDA=2
# export LR=3e-5
export CUDA=1
export LR=3e-5
export CUSTOM_NAME=ft_probe_v1
export MODEL_NAME=bart-large
export epoch=20
export OUTPUT_DIR=${PWD}/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}.lr${LR}.bz16
export DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order_probe

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=facebook/$MODEL_NAME \
--learning_rate=${LR} \
--train_batch_size=2 \
--eval_batch_size=2 \
--gradient_accumulation_steps 8 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=${epoch}  \
--gpus=1 \
--do_predict \
--max_source_length 512 \
--max_target_length 250 \
--val_max_target_length 250 \
--eval_max_gen_length 250 \
--test_max_target_length 250 \
--overwrite_output_dir 






echo "CLASSIFICATION LOSS + PROBE"

export CUDA=0
export TASK=lr_vs_roundup #propaganda
export ratio=1

export MODEL_NAME=bart-large
export epoch=20
export DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order_probe # version 1
export CUSTOM_NAME=ft_probe_v1_mt_${ratio}_${TASK}

export LR=3e-5
export OUTPUT_DIR=${PWD}/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}.lr${LR}.bz16

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=facebook/$MODEL_NAME \
--learning_rate=${LR} \
--train_batch_size=1 \
--eval_batch_size=1 \
--gradient_accumulation_steps 16 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=${epoch}  \
--gpus=1 \
--do_train "$@" \
--do_predict \
--max_source_length 512 \
--max_target_length 250 \
--val_max_target_length 250 \
--eval_max_gen_length 250 \
--test_max_target_length 250 \
--overwrite_output_dir \
--extra_task $TASK 



'''############## TEST #############'''
export CUDA=1
export LR=3e-5
export CUSTOM_NAME=ft_probe_v2
export MODEL_NAME=bart-large
export epoch=20
export DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order_probe_v2_pred_titles
export SAVE_MODEL_PATH=/home/nayeon/neutralization/src/naacl2022/bart-large.ft_probe.lr3e-5.bz16

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=facebook/$MODEL_NAME \
--learning_rate=${LR} \
--train_batch_size=2 \
--eval_batch_size=2 \
--gradient_accumulation_steps 8 \
--output_dir=${SAVE_MODEL_PATH} \
--num_train_epochs=${epoch}  \
--gpus=1 \
--do_predict \
--max_source_length 512 \
--max_target_length 250 \
--val_max_target_length 250 \
--eval_max_gen_length 250 \
--test_max_target_length 250 \
--overwrite_output_dir 






echo "================== BART - MultiNews ================== "

############# TRAIN ##############
export CUDA=7
export DATA_DIR=/home/nayeon/neutralization/data/multi-news
export BSZ=2
export ACC_STEP=8
export LR=3e-5
export EPOCH=10 # 20
export CUSTOM_NAME=multinews_ft
export MODEL_NAME=bart-large
export OUTPUT_DIR=${PWD}/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}.${LR}.512_250.e${EPOCH}
# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=facebook/$MODEL_NAME \
--learning_rate=${LR} \
--train_batch_size=${BSZ} \
--eval_batch_size=${BSZ} \
--gradient_accumulation_steps ${ACC_STEP} \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=${EPOCH}  \
--gpus=1 \
--do_train "$@" \
--do_predict \
--max_source_length 512 \
--max_target_length 250 \
--val_max_target_length 250 \
--eval_max_gen_length 250 \
--test_max_target_length 250 \
--overwrite_output_dir 


############# TEST ##############
export CUDA=7
# export DATA_DIR=/home/nayeon/neutralization/data/multi-news
export DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order
export BSZ=2
export ACC_STEP=8
export LR=3e-5
export EPOCH=10 # 20
export CUSTOM_NAME=multinews_ft
export MODEL_NAME=bart-large
export OUTPUT_DIR=${PWD}/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}.${LR}.512_250.e${EPOCH}
# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=facebook/$MODEL_NAME \
--learning_rate=${LR} \
--train_batch_size=${BSZ} \
--eval_batch_size=${BSZ} \
--gradient_accumulation_steps ${ACC_STEP} \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=${EPOCH}  \
--gpus=1 \
--do_predict \
--max_source_length 512 \
--max_target_length 250 \
--val_max_target_length 250 \
--eval_max_gen_length 250 \
--test_max_target_length 250 \
--overwrite_output_dir 
