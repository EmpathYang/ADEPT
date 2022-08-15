model_type=$1
bias=$2
data=$3
gpu=$4
block_size=128

if [ $model_type = 'bert' ]; then
    model_name_or_path=bert-large-uncased
elif [ $model_type = 'roberta' ]; then
    model_name_or_path=roberta-large
fi

if [ $bias -eq 'gender']; then
    attribute_words='../data/female_seat.txt,../data/male_seat.txt'
elif [$bias -eq 'religion']; then
    attribute_words='../data/jewish.txt,../data/christian.txt,../data/muslim.txt'
fi

OUTPUT_DIR=../sentences_collection/$model_name_or_path/$bias/word_correlation

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

python -u ../preprocess_plot_word_correlation.py --input ../data/$data \
                        --stereotypes ../data/neutral_seat.txt \
                        --attributes $attribute_words \
                        --output $OUTPUT_DIR \
                        --block_size $block_size \
                        --model_type $model_type

