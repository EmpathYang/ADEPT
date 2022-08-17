# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import os
import random
from typing import List, Tuple
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from utils import plot_word_correlation_gender, plot_word_correlation_religion
from model.sequence_classification import BertPrefixForSequenceClassification

from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

def get_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_file", default=None, type=str, required=True, help="The input data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--bias", default="gender", type=str, choices=["gender", "religion"], help="Bias type.")
    parser.add_argument("--tuning_type", default="prompt_tuning", type=str, choices=["prompt_tuning", "finetuning"], help="Tuning type")
    parser.add_argument("--model_name_or_path", default=None, type=str, help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.")
    parser.add_argument("--config_name", default=None, type=str, help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.")
    parser.add_argument("--tokenizer_name", default=None, type=str, help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.")
    parser.add_argument("--cache_dir", default=None, type=str, help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)")
    parser.add_argument("--block_size", default=-1, type=int, help="Optional input sequence length after tokenization." "The training dataset will be truncated in block of this size for training." "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size per vector.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--perplexity", type=float, default=1.0, help="Set perplecity(hyperparameter).")
    parser.add_argument("--threshold", default=30, type=int, help="Set threshold for LNL.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    return args

class LineByLineTextDataset(Dataset):
    def __init__(self, examples: list, labels: list):
        self.examples = examples
        self.labels = labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.labels:
            return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)
        else:
            return torch.tensor(self.examples[i], dtype=torch.long)

def create_dataset(data, dataset):
    words_list = data['words_list']
    words_examples = data['words_examples']
    words_indexes = data['words_indexes']
    d = dict()
    for word, example, index in zip(words_list, words_examples, words_indexes):
        d[word] = dataset(example, index)
    return d

def filter_data(data, threshold):
    _words_list = data['words_list']
    _words_examples = data['words_examples']
    _words_indexes = data['words_indexes']
    words_list = []
    words_examples = []
    words_indexes = []
    for word, _example, _index in zip(_words_list, _words_examples, _words_indexes):
        if len(_example) >= threshold:
            words_list.append(word)
            chosen_example = random.choices(range(len(_example)), k=threshold)
            example = []
            index = []
            for i in chosen_example:
                example.append(_example[i])
                index.append(_index[i])
            words_examples.append(example)
            words_indexes.append(index)
    return {'words_list': words_list,
            'words_examples': words_examples,
            'words_indexes': words_indexes}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def create_dataloader(args, datasets, tokenizer):
    def collate(batch: List[torch.Tensor]):
        if type(batch[0]) == tuple:
            examples, labels = list(zip(*batch))
            padded_examples = pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
            examples_attention_mask = torch.zeros_like(padded_examples, dtype=torch.int32)
            examples_attention_mask[torch.where(padded_examples != tokenizer.pad_token_id)] = 1
            padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
            labels_attention_mask = torch.zeros_like(padded_labels, dtype=torch.int32)
            labels_attention_mask[torch.where(padded_labels != 0)] = 1
            return padded_examples, padded_labels, examples_attention_mask, labels_attention_mask
        else:
            padded_examples = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
            examples_attention_mask = torch.zeros_like(padded_examples, dtype=torch.int32)
            examples_attention_mask[torch.where(padded_examples != tokenizer.pad_token_id)] = 1
            return padded_examples, examples_attention_mask

    dataloaders = {}
    example_num = 0
    word_list = []

    for key, dataset in datasets.items():
        example_num += len(dataset)
        dataloaders[key] = iter(DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=False))
        word_list += [key for _ in range(int(args.threshold / args.batch_size))]

    return dataloaders, example_num, word_list

def calculate(args, data, datasets, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:

    dataloaders, _, word_list = create_dataloader(args, datasets, tokenizer)

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    model.eval()

    def get_hiddens_of_model(input, input_attention_mask):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        with torch.no_grad():
            if args.model_type == 'roberta':
                if args.tuning_type == 'prompt_tuning':
                    hiddens = model(input, input_attention_mask).hidden_states
                elif args.tuning_type == 'finetuning':
                    hiddens = model.roberta(input).hidden_states
            elif args.model_type == 'bert':
                if args.tuning_type == 'prompt_tuning':
                    hiddens = model(input, input_attention_mask).hidden_states
                elif args.tuning_type == 'finetuning':
                    hiddens = model.bert(input).hidden_states

        return hiddens

    words_hiddens = {word: [] for word in word_list}
    for word in word_list:
        inputs, labels, inputs_attention_mask, _ = next(dataloaders[word])
        inputs = inputs.to(args.device)
        inputs_attention_mask = inputs_attention_mask.to(args.device)
        hiddens = get_hiddens_of_model(inputs, inputs_attention_mask)
        hiddens = torch.stack(hiddens, 2)
        if labels.size(1) > 1:
            onehot = torch.eye(hiddens.size(1))
            zeros = torch.zeros(1, onehot.size(0))
            onehot = torch.cat((zeros, onehot), 0)
            onehot = onehot[labels]
            onehot = torch.sum(onehot, 1)
            onehot = onehot.view(hiddens.size(0), -1, 1, 1)
        else:
            onehot = torch.eye(hiddens.size(1))[labels].view(hiddens.size(0), -1, 1, 1)
        onehot = onehot.to(args.device)
        words_hiddens[word].append(torch.sum(hiddens * onehot, 1) / labels.size(1))

    for word in data['words_list']:
        words_hiddens[word] = torch.mean(torch.cat(words_hiddens[word], 0), 0).detach()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(words_hiddens, args.output_dir + f'/word_embedding_{args.threshold}.bin')

def main():
    
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    config.output_hidden_states = 'true'

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        try:
            args.block_size = min(args.block_size, tokenizer.model_max_length)
        except:
            args.block_size = min(args.block_size, tokenizer.max_len)

    if args.tuning_type == "prompt_tuning":
        model = BertPrefixForSequenceClassification.from_pretrained(args.model_name_or_path)
    elif args.tuning_type == "finetuning":
        if args.model_name_or_path:
            model = AutoModelWithLMHead.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )
        else:
            model = AutoModelWithLMHead.from_config(config)

    # GPT-2 and GPT do not have pad.
    if tokenizer._pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    data = torch.load(args.data_file)
    data = filter_data(data, args.threshold)

    datasets = create_dataset(data, LineByLineTextDataset)

    calculate(args, data, datasets, model, tokenizer)

    layers = [[24]]
    for l in layers:
        if args.bias == 'gender':
            plot_word_correlation_gender(args.output_dir, layers=l, perplexity=30)
        elif args.bias == 'religion':
            plot_word_correlation_religion(args.output_dir, layers=l, perplexity=80)


if __name__ == "__main__":
    main()
