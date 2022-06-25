
export CUDA=2
export LR=3e-5
export CUSTOM_NAME=ft_probe_v1
export MODEL_NAME=bart-large
export epoch=20

export PROJECT_DIR=/home/nayeon/neutralization
export OUTPUT_DIR=$PROJECT_DIR/src/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}.lr${LR}.bz16
export DATA_DIR=$PROJECT_DIR/data/naacl2022_lrc_roundup_random_order_probe

CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
--data_dir=$DATA_DIR \
--model_name_or_path=facebook/$MODEL_NAME \
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