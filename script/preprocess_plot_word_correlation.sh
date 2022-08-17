model_type=$1
gpu=$2
bias=$3
data=$4
block_size=128

if [ $model_type == 'bert' ]; then
    model_name_or_path=bert-large-uncased
elif [ $model_type == 'roberta' ]; then
    model_name_or_path=roberta-large
fi

if [ $bias == 'gender' ]; then
    attribute_words='../data/female.txt,../data/male.txt'
elif [ $bias == 'religion' ]; then
    attribute_words='../data/judaism.txt,../data/christianity.txt,../data/islam.txt'
fi

OUTPUT_DIR=../sentence_collection/$model_name_or_path/$bias/word_correlation

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

python -u ../preprocess_plot_word_correlation.py --input ../data/$data \
                        --neutral_words ../data/neutral.txt \
                        --attribute_words $attribute_words \
                        --output $OUTPUT_DIR \
                        --block_size $block_size \
                        --model_type $model_type

