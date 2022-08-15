model_type=$1
gpu=$2
tuning_type=$3 # prompt_tuning finetuning
bias=$4 # gender religion
model_name_or_path=$5
seed=42

if [ $model_type = 'bert' ]; then
    local_model_name_or_path=bert-large-uncased
elif [ $model_type = 'roberta' ]; then
    local_model_name_or_path=roberta-large
fi

PLOT_DATA=../sentences_collection/$local_model_name_or_path/$bias/word_correlation/word_data.bin
OUTPUT_DIR=../debiased_models/$seed/$local_model_name_or_path/$algorithm/$bias

rm -r $OUTPUT_DIR

echo $model_type $tag $bias $gpu $seed

CUDA_VISIBLE_DEVICES=$gpu python ../plot_word_correlation.py \
    --bias $bias \
    --tuning_type $tuning_type \
    --data_file=$PLOT_DATA \
    --model_type=$model_type \
    --model_name_or_path=$model_name_or_path \
    --output_dir=$OUTPUT_DIR \
    --threshold 30 \
    --perplexity 30 \
    --batch_size 2 \
    --block_size 128 \
    --seed $seed \