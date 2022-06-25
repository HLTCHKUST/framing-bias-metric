# Script for verifying that run_bart_sum can be invoked from its directory

# Get tiny dataset with cnn_dm format (4 examples for train, val, test)
# wget https://cdn-datasets.huggingface.co/summarization/cnn_tiny.tgz
# tar -xzvf cnn_tiny.tgz
# rm cnn_tiny.tgz

export OUTPUT_DIR_NAME=bart_base
export CURRENT_DIR=${PWD}/output_body
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# --model_name_or_path=facebook/bart-large-cnn \

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and testing_utils.py
# --model_name_or_path=sshleifer/bart-tiny-random \
export PYTHONPATH="../":"${PYTHONPATH}"
CUDA_VISIBLE_DEVICES=1 python finetune.py \
--data_dir=/home/nayeon/omission/data/body \
--model_name_or_path=facebook/bart-base \
--learning_rate=3e-5 \
--train_batch_size=8 \
--eval_batch_size=8 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=2  \
--gpus=1 \
--do_train "$@" \
--do_predict \
--overwrite_output_dir

# rm -rf cnn_tiny
# rm -rf $OUTPUT_DIR