import argparse
import regex as re
import nltk

import torch
import csv

from transformers import BertTokenizer, RobertaTokenizer
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    tp = lambda x:list(x.split(','))

    parser.add_argument('--input', type=str, required=True,
                        help='Data')
    parser.add_argument('--neutral_words', type=str)
    parser.add_argument('--attribute_words', type=tp, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['bert', 'roberta'])

    args = parser.parse_args()

    return args

def prepare_tokenizer(args):
    if args.model_type == 'bert':
        pretrained_weights = 'bert-large-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'roberta':
        pretrained_weights = 'roberta-large'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
    return tokenizer

def main(args):
    data = [l.strip() for l in open(args.input)]
    stereotypes = [word.strip() for word in open(args.neutral_words)]
    stereotype_set = set(stereotypes)

    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    sequential_l = []
    attributes_l = []
    all_attributes_set = set()
    for attribute in args.attribute_words:
        l = [word.strip() for word in open(attribute)]
        sequential_l.append(l)
        attributes_l.append(set(l))
        all_attributes_set |= set(l)
    words_list = list(all_attributes_set | stereotype_set)

    tokenizer = prepare_tokenizer(args)

    words_examples = [[] for _ in range(len(words_list))]
    words_indexes = [[] for _ in range(len(words_list))]

    for line in tqdm(data):
        neutral_flag = True
        line = line.strip()
        if len(line) < 1:
            continue
        leng = len(line.split())
        if leng > args.block_size or leng <= 1:
            continue
        tokens_orig = [token.strip() for token in re.findall(pat, line)]
        tokens_lower = [token.lower() for token in tokens_orig]
        token_set = set(tokens_lower)

        attribute_other_l = []
        for i, _ in enumerate(attributes_l):
            a_set = set()
            for j, attribute in enumerate(attributes_l):
                if i != j:
                    a_set |= attribute
            attribute_other_l.append(a_set)

        for i, (attribute_set, other_set) in enumerate(zip(attributes_l, attribute_other_l)):
            if attribute_set & token_set:
                neutral_flag = False
                if not other_set & token_set:
                    orig_line = line
                    line = tokenizer.encode(line, add_special_tokens=True)
                    labels = attribute_set & token_set
                    for ori_label in list(labels):
                        idx = tokens_lower.index(ori_label)
                        label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True))[1:-1]
                        line_ngram = list(nltk.ngrams(line, len(label)))
                        try:
                            idx = line_ngram.index(label)
                            index = words_list.index(ori_label)
                            words_examples[index].append(line)
                            words_indexes[index].append([idx + j for j in range(len(label))])
                        except:
                            print('unwanted labels', orig_line, labels)
                break

        if neutral_flag:
            if stereotype_set & token_set:
                orig_line = line
                line = tokenizer.encode(line, add_special_tokens=True)
                labels = stereotype_set & token_set
                for ori_label in list(labels):
                    idx = tokens_lower.index(ori_label)
                    label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True))[1:-1]
                    line_ngram = list(nltk.ngrams(line, len(label)))
                    try:
                        idx = line_ngram.index(label)
                        index = words_list.index(ori_label)
                        words_examples[index].append(line)
                        words_indexes[index].append([idx + j for j in range(len(label))])
                    except:
                        print('unwanted labels', orig_line, labels)

    with open(args.output + '/word_sentence_number.csv', 'w') as wf:
        writer = csv.writer(wf)
        writer.writerow(['Word', 'Sentence Number'])
        for _word, _sentence, _ in zip(words_list, words_examples, words_indexes):
            writer.writerow([_word, len(_sentence)])

    check_count = {}
    words_examples_buffer = [[] for _ in range(len(words_list))]
    words_indexes_buffer = [[] for _ in range(len(words_list))]
    for attributes in zip(*(sequential_l)):
        try:
            min_size = min([int(len(words_examples[words_list.index(a)])) for a in attributes])
            if min_size < 30:
                continue
            for i, a in enumerate(attributes):
                check_count[a] = min_size
                words_examples_buffer[words_list.index(a)] += words_examples[words_list.index(a)][:min_size]
                words_indexes_buffer[words_list.index(a)] += words_indexes[words_list.index(a)][:min_size]
        except:
            continue
    for n in stereotype_set:
        words_examples_buffer[words_list.index(n)] += words_examples[words_list.index(n)]
        words_indexes_buffer[words_list.index(n)] += words_indexes[words_list.index(n)]
    words_examples = words_examples_buffer
    words_indexes = words_indexes_buffer
    print(check_count)

    data = {'words_list': words_list,
            'words_examples': words_examples,
            'words_indexes': words_indexes}

    torch.save(data, args.output + '/word_data.bin')

if __name__ == "__main__":
    args = get_args()
    main(args)

