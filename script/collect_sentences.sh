model_type=$1
data=$2
bias=$3
ab_test=$4
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

AB_TEST_TYPE_LIST=(
    'raw' 
    'reliability' 
    'quality' 
    'quantity-100' 
    'quantity-1000' 
    'quantity-10000' 
    'final'
    )

if [ $ab_test != 'all' ]; then
    OUTPUT_DIR=../sentences_collection/$model_name_or_path/$bias/$ab_test

    rm -r $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR
    echo $model_type $bias $ab_test

    python -u ../collect_sentences.py --input ../data/$data \
                            --neutral_words ../data/neutral.txt \
                            --attribute_words $attribute_words \
                            --output $OUTPUT_DIR \
                            --block_size $block_size \
                            --model_type $model_type \
                            --ab_test_type $ab_test \

else
    for ab_test_type in ${AB_TEST_TYPE_LIST[@]}; do
        OUTPUT_DIR=../sentence_collection/$model_name_or_path/$bias/$ab_test_type

        rm -r $OUTPUT_DIR
        mkdir -p $OUTPUT_DIR
        echo $model_type $bias $ab_test_type

        python -u ../collect_sentences.py --input ../data/$data \
                                --neutral_words ../data/neutral.txt \
                                --attribute_words $attribute_words \
                                --output $OUTPUT_DIR \
                                --block_size $block_size \
                                --model_type $model_type \
                                --ab_test_type $ab_test_type \

    done
fi