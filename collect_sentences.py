import argparse
import regex as re
import nltk
import torch
from transformers import *
import random

def parse_args():
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
    parser.add_argument('--ab_test_type', type=str, default='final',
                        choices=['raw', 'reliability', 'quality', 'quantity-100', 'quantity-1000', 'quantity-10000', 'final'])

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
    SUPERVISED_ENTITIES = ['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna', 'John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill', 'executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career', 'home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives', 'math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition', 'poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture', 'science', 'technology', 'physics', 'chemistry', 'Einstein', 'NASA', 'experiment', 'astronomy', 'Shakespeare']
    supervised_entities = [w.lower() for w in SUPERVISED_ENTITIES]
    entity_count = {}

    data = [l.strip() for l in open(args.input)]
    if args.neutral_words:
        neutrals = [word.strip() for word in open(args.neutral_words)]
        neutral_set = set(neutrals)

    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    sequential_l = []
    attributes_l = []
    all_attributes_set = set()
    for attribute in args.attributes:
        l = [word.strip() for word in open(attribute)]
        sequential_l.append(l)
        attributes_l.append(set(l))
        all_attributes_set |= set(l)

    tokenizer = prepare_tokenizer(args)

    neutral_examples = []
    if args.neutral_words:
        neutral_labels = []
    attributes_examples = [{} for _ in range(len(attributes_l))]
    attributes_labels = [{} for _ in range(len(attributes_l))]

    for orig_line in data:
        neutral_flag = True
        orig_line = orig_line.strip()
        if len(orig_line) < 1:
            continue
        leng = len(orig_line.split())
        if leng > args.block_size or leng <= 1:
            continue
        tokens_orig = [token.strip() for token in re.findall(pat, orig_line)]
        tokens_lower = [token.lower() for token in tokens_orig]
        token_set = set(tokens_lower)

        for i, attribute_set in enumerate(attributes_l):
            if attribute_set & token_set:
                neutral_flag = False
                line = tokenizer.encode(orig_line, add_special_tokens=True)
                labels = attribute_set & token_set
                for ori_label in list(labels):
                    if ori_label in SUPERVISED_ENTITIES + supervised_entities:
                        print(f'{ori_label}: {orig_line}')
                    if ori_label in entity_count.keys():
                        entity_count[ori_label] += 1
                    else:
                        entity_count[ori_label] = 1
                    idx = tokens_lower.index(ori_label)
                    label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True))[1:-1]
                    line_ngram = list(nltk.ngrams(line, len(label)))
                    try:
                        idx = line_ngram.index(label)
                        if ori_label in attributes_examples[i].keys():
                            attributes_examples[i][ori_label].append(line)
                            attributes_labels[i][ori_label].append([idx + j for j in range(len(label))])
                        else:
                            attributes_examples[i][ori_label] = [line]
                            attributes_labels[i][ori_label] = [[idx + j for j in range(len(label))]]
                    except:
                        pass

        if neutral_flag:
            if args.neutral_words:
                if neutral_set & token_set:
                    line = tokenizer.encode(orig_line, add_special_tokens=True)
                    labels = neutral_set & token_set
                    for label in list(labels):
                        if label in SUPERVISED_ENTITIES + supervised_entities:
                            print(f'{label}: {orig_line}')
                        if label in entity_count.keys():
                            entity_count[label] += 1
                        else:
                            entity_count[label] = 1
                        idx = tokens_lower.index(label)
                        label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True))[1:-1]
                        line_ngram = list(nltk.ngrams(line, len(label)))
                        try:
                            idx = line_ngram.index(label)
                            neutral_examples.append(line)
                            neutral_labels.append([idx + i for i in range(len(label))])
                        except:
                            pass
            else:
                neutral_examples.append(tokenizer.encode(line, add_special_tokens=True))

    attributes_examples_buffer = [[] for _ in range(len(attributes_l))]
    attributes_labels_buffer = [[] for _ in range(len(attributes_l))]
    for attributes in zip(*(sequential_l)):
        try:
            if args.ab_test_type == 'raw' or args.ab_test_type == 'quantity-100' or args.ab_test_type == 'quantity-1000' or args.ab_test_type == 'quantity-10000':
                for i, a in enumerate(attributes):
                    attributes_examples_buffer[i] += attributes_examples[i][a][:]
                    attributes_labels_buffer[i] += attributes_labels[i][a][:]
            elif args.ab_test_type == 'reliability':
                for i, a in enumerate(attributes):
                    if len(attributes_examples[i][a]) < 30:
                        continue
                    attributes_examples_buffer[i] += attributes_examples[i][a][:]
                    attributes_labels_buffer[i] += attributes_labels[i][a][:]
            elif args.ab_test_type == 'quality':
                min_size = min([len(attributes_examples[i][a]) for i, a in enumerate(attributes)])
                for i, a in enumerate(attributes):
                    attributes_examples_buffer[i] += attributes_examples[i][a][:min_size]
                    attributes_labels_buffer[i] += attributes_labels[i][a][:min_size]
            elif args.ab_test_type == 'final':
                min_size = min([len(attributes_examples[i][a]) for i, a in enumerate(attributes)])
                if min_size < 30:
                    continue
                for i, a in enumerate(attributes):
                    attributes_examples_buffer[i] += attributes_examples[i][a][:min_size]
                    attributes_labels_buffer[i] += attributes_labels[i][a][:min_size]
            else:
                raise Exception()
        except:
            continue
    attributes_examples = attributes_examples_buffer
    attributes_labels = attributes_labels_buffer

    if args.ab_test_type == 'quantity-100' or args.ab_test_type == 'quantity-1000' or args.ab_test_type == 'quantity-10000':
        attributes_examples_buffer = [[] for _ in range(len(attributes_l))]
        attributes_labels_buffer = [[] for _ in range(len(attributes_l))]
        min_size = min([len(attributes_examples[i]) for i in range(len(attributes_l))])
        if args.ab_test_type == 'quantity-100':
            chosen_index = random.choices(range(min_size), k=1100)
        elif args.ab_test_type == 'quantity-1000':
            chosen_index = random.choices(range(min_size), k=2000)
        elif args.ab_test_type == 'quantity-10000':
            chosen_index = random.choices(range(min_size), k=11000)
        for i in range(len(attributes_l)):
            for j in chosen_index:
                attributes_examples_buffer[i].append(attributes_examples[i][j])
                attributes_labels_buffer[i].append(attributes_labels[i][j])
        attributes_examples = attributes_examples_buffer
        attributes_labels = attributes_labels_buffer

    with open(args.output + '/count.txt', 'a') as wf:
        print(entity_count, file=wf)
        print('neutral:', len(neutral_examples), file=wf)
        for i, examples in enumerate(attributes_examples):
            print(f'attributes{i}:', len(examples), file=wf)

    data = {'attributes_examples': attributes_examples,
            'attributes_labels': attributes_labels,
            'neutral_examples': neutral_examples}

    if args.neutral_words:
        data['neutral_labels'] = neutral_labels

    torch.save(data, args.output + '/data.bin')

if __name__ == "__main__":
    args = parse_args()
    main(args)
