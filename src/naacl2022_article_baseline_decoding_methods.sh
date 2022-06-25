
export PYTHONPATH="../":"${PYTHONPATH}"

echo "================== TEST PHASE ONLY ============== ARTICLE BASELINE ================== "

echo "[Default: BEAM 4] FT with probestyle"
export CUDA=7
export LR=3e-5
export CUSTOM_NAME=ft_probe_v1
export MODEL_NAME=bart-large
export epoch=20
export OUTPUT_DIR=${PWD}/naacl2022/${MODEL_NAME}.${CUSTOM_NAME}.lr${LR}.bz16
export DATA_DIR=/home/nayeon/neutralization/data/naacl2022_lrc_roundup_random_order_probe

# CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
# --data_dir=$DATA_DIR \
# --model_name_or_path=facebook/$MODEL_NAME \
# --eval_batch_size=2 \
# --output_dir=$OUTPUT_DIR \
# --gpus=1 \
# --do_predict \
# --max_source_length 512 \
# --max_target_length 250 \
# --val_max_target_length 250 \
# --eval_max_gen_length 250 \
# --test_max_target_length 250 \
# --num_return_sequences 4 \
# --custom_pred_file_suffix 4candidate \
# --overwrite_output_dir 



echo "top-p"
for p in 0.9 0.8 0.5
do
    export CUDA=1
    export temp=1.0
    CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
    --data_dir=$DATA_DIR \
    --model_name_or_path=facebook/$MODEL_NAME \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --output_dir=$OUTPUT_DIR \
    --gpus=1 \
    --do_predict \
    --max_source_length 512 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --eval_max_gen_length 250 \
    --test_max_target_length 250 \
    --eval_beams 4 \
    --do_sample \
    --top_p ${p} \
    --temperature ${temp} \
    --custom_pred_file_suffix top-p-${p}_temp-${temp} \
    --overwrite_output_dir 

    # export CUDA=0
    export temp=0.9
    CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
    --data_dir=$DATA_DIR \
    --model_name_or_path=facebook/$MODEL_NAME \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --output_dir=$OUTPUT_DIR \
    --gpus=1 \
    --do_predict \
    --max_source_length 512 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --eval_max_gen_length 250 \
    --test_max_target_length 250 \
    --eval_beams 4 \
    --do_sample \
    --top_p ${p} \
    --temperature ${temp} \
    --custom_pred_file_suffix top-p-${p}_temp-${temp} \
    --overwrite_output_dir 

    # export CUDA=2
    export temp=0.5
    CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
    --data_dir=$DATA_DIR \
    --model_name_or_path=facebook/$MODEL_NAME \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --output_dir=$OUTPUT_DIR \
    --gpus=1 \
    --do_predict \
    --max_source_length 512 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --eval_max_gen_length 250 \
    --test_max_target_length 250 \
    --eval_beams 4 \
    --do_sample \
    --top_p ${p} \
    --temperature ${temp} \
    --custom_pred_file_suffix top-p-${p}_temp-${temp} \
    --overwrite_output_dir 

    # export CUDA=7
    export temp=0.1
    CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
    --data_dir=$DATA_DIR \
    --model_name_or_path=facebook/$MODEL_NAME \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --output_dir=$OUTPUT_DIR \
    --gpus=1 \
    --do_predict \
    --max_source_length 512 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --eval_max_gen_length 250 \
    --test_max_target_length 250 \
    --eval_beams 4 \
    --do_sample \
    --top_p ${p} \
    --temperature ${temp} \
    --custom_pred_file_suffix top-p-${p}_temp-${temp} \
    --overwrite_output_dir 
done



echo "top-k"

for k in 5 10 15 50
do 
    export CUDA=1
    export temp=1.0
    CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
    --data_dir=$DATA_DIR \
    --model_name_or_path=facebook/$MODEL_NAME \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --output_dir=$OUTPUT_DIR \
    --gpus=1 \
    --do_predict \
    --max_source_length 512 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --eval_max_gen_length 250 \
    --test_max_target_length 250 \
    --do_sample \
    --top_k ${k} \
    --temperature ${temp} \
    --custom_pred_file_suffix top-k-${k}_temp-${temp} \
    --overwrite_output_dir &

    # export CUDA=0
    export temp=0.9
    CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
    --data_dir=$DATA_DIR \
    --model_name_or_path=facebook/$MODEL_NAME \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --output_dir=$OUTPUT_DIR \
    --gpus=1 \
    --do_predict \
    --max_source_length 512 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --eval_max_gen_length 250 \
    --test_max_target_length 250 \
    --do_sample \
    --top_k ${k} \
    --temperature ${temp} \
    --custom_pred_file_suffix top-k-${k}_temp-${temp} \
    --overwrite_output_dir 

    # export CUDA=2
    export temp=0.5
    CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
    --data_dir=$DATA_DIR \
    --model_name_or_path=facebook/$MODEL_NAME \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --output_dir=$OUTPUT_DIR \
    --gpus=1 \
    --do_predict \
    --max_source_length 512 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --eval_max_gen_length 250 \
    --test_max_target_length 250 \
    --do_sample \
    --top_k ${k} \
    --temperature ${temp} \
    --custom_pred_file_suffix top-k-${k}_temp-${temp} \
    --overwrite_output_dir &


    # export CUDA=7
    export temp=0.1
    CUDA_VISIBLE_DEVICES=$CUDA python finetune.py \
    --data_dir=$DATA_DIR \
    --model_name_or_path=facebook/$MODEL_NAME \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --output_dir=$OUTPUT_DIR \
    --gpus=1 \
    --do_predict \
    --max_source_length 512 \
    --max_target_length 250 \
    --val_max_target_length 250 \
    --eval_max_gen_length 250 \
    --test_max_target_length 250 \
    --do_sample \
    --top_k ${k} \
    --temperature ${temp} \
    --custom_pred_file_suffix top-k-${k}_temp-${temp} \
    --overwrite_output_dir
done


