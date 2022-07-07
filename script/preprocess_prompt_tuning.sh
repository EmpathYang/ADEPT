model_type=$1
data=$2
seed=42
block_size=128

if [ $model_type = 'bert' ]; then
    model_name_or_path=bert-large-uncased
elif [ $model_type = 'roberta' ]; then
    model_name_or_path=roberta-large
elif [ $model_type = 'albert' ]; then
    model_name_or_path=albert-base-v2
elif [ $model_type = 'dbert' ]; then
    model_name_or_path=distilbert-base-uncased
elif [ $model_type = 'electra' ]; then
    model_name_or_path=google/electra-small-discriminator
fi

OUTPUT_DIR=../preprocess/$seed/$model_name_or_path

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

python -u ../preprocess_prompt_tuning.py --input ../data/$data \
                        --stereotypes ../data/stereotype.txt \
                        --attributes ../data/female.txt,../data/male.txt \
                        --output $OUTPUT_DIR \
                        --seed $seed \
                        --block_size $block_size \
                        --model_type $model_type

