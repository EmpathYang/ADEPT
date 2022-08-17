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
import logging
import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
import sys
import time
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from model.utils import get_model, TaskType
from arguments import get_args
from distance import  KL_divergence, calculate_group_to_one_relative_distance_asymmetric, JS_divergence

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
timestr = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

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
    d = dict()
    for key in data['example'].keys():
        if key not in data['label']:
            d[key] = dataset(data['example'][key], None)
        else:
            d[key] = dataset(data['example'][key], data['label'][key])

    return d

def load_and_cache_examples(data, args, tokenizer):
    train_dataset = create_dataset(data['train'], LineByLineTextDataset)
    dev_dataset = create_dataset(data['dev'], LineByLineTextDataset)
    return {'train': train_dataset, 'dev': dev_dataset}

def split_data(attributes_examples, attributes_labels, neutral_examples, neutral_labels, args):
    data = {'train': {'example': {}, 'label': {}}, 'dev': {'example': {}, 'label': {}}}

    for i, (examples, labels) in enumerate(zip(attributes_examples, attributes_labels)):
        idx_l = list(range(len(examples)))
        random.shuffle(idx_l)
        examples = [examples[idx] for idx in idx_l]
        labels = [labels[idx] for idx in idx_l]
        data['train']['example'][f'attribute{i}'] = examples[args.dev_data_size:]
        data['train']['label'][f'attribute{i}'] = labels[args.dev_data_size:]
        data['dev']['example'][f'attribute{i}'] = examples[:args.dev_data_size]
        data['dev']['label'][f'attribute{i}'] = labels[:args.dev_data_size]

    idx_l = list(range(len(neutral_examples)))
    random.shuffle(idx_l)
    neutral_examples = [neutral_examples[idx] for idx in idx_l]
    data['train']['example']['neutral'] = neutral_examples[args.dev_data_size:]
    data['dev']['example']['neutral'] = neutral_examples[:args.dev_data_size]
    if neutral_labels is not None:
        neutral_labels = [neutral_labels[idx] for idx in idx_l]
        data['train']['label']['neutral'] = neutral_labels[args.dev_data_size:]
        data['dev']['label']['neutral'] = neutral_labels[:args.dev_data_size]

    return data

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def create_dataloader(args, datasets, tokenizer, train=False):
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
    data_distribution = []

    min_size = min([len(value) for key, value in datasets.items() if key != 'neutral'])

    for key, dataset in datasets.items():
        example_num += len(dataset)
        if train:
            dataloaders[key] = iter(DataLoader(dataset, batch_size=args.train_batch_size, collate_fn=collate, shuffle=True))
            data_distribution += [key for _ in range(int(min_size / args.train_batch_size))]
        else:
            dataloaders[key] = iter(DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=collate , shuffle=False))
            data_distribution += [key for _ in range(int(min_size / args.eval_batch_size))]

    return dataloaders, example_num, data_distribution

def train(args, data, datasets, model: PreTrainedModel, original_model, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)

    train_datasets = datasets['train']
    dev_datasets = datasets['dev']

    train_dataloaders, train_example_num, train_distribution = create_dataloader(args, train_datasets, tokenizer, train=True)
    dev_dataloaders, dev_example_num, dev_distribution = create_dataloader(args, dev_datasets, tokenizer, train=False)

    train_iter_num = sum([len(dataloader) for dataloader in train_dataloaders.values()])

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_iter_num // args.gradient_accumulation_steps) + 1
    else:
        t_total = train_iter_num // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    original_model = original_model.module if hasattr(original_model, "module") else original_model  # Take care of distributed/parallel training
    original_model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        original_model = torch.nn.DataParallel(original_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        original_model = torch.nn.parallel.DistributedDataParallel(
            original_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("Num examples = %d", train_example_num)
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info(
        "Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_loss = float('inf')
    best_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (train_iter_num // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (train_iter_num // args.gradient_accumulation_steps)

            logger.info("Continuing training from checkpoint, will skip to saved global_step")
            logger.info("Continuing training from epoch %d", epochs_trained)
            logger.info("Continuing training from global step %d", global_step)
            logger.info("Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("Starting prompt-tuning.")


    model.zero_grad()
    original_model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    def inner_product(x, y):
        return torch.mean(torch.sum(y * x, 3))

    def mean_square(x, y, idx):
        return torch.mean(torch.mean((y - x) ** 2, idx))

    def save_best_model(best_loss, best_step, dev_dataloaders):
        if (args.local_rank == -1):  # Only evaluate when single GPU otherwise metrics may not average well
            eval_loss = evaluate(model, attributes_hiddens, dev_dataloaders)
            logger.info("global_step = %s, evaluate loss = %s", global_step, eval_loss)
            tb_writer.add_scalar("eval_loss", eval_loss, global_step)
        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_step = global_step
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, timestr, 'best_model_ckpt')
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("best_step = %s, best loss = %s", best_step, best_loss)

        return best_loss, best_step

    def get_hiddens_of_model(input, input_attention_mask):
        model.zero_grad()
        if args.model_type == 'roberta':
            if args.algorithm == 'ADEPT':
                hiddens = model(input, input_attention_mask).hidden_states
            elif args.algorithm == 'ADEPT-finetuning' or args.algorithm == 'DPCE':
                hiddens = model.roberta(input).hidden_states
        elif args.model_type == 'bert':
            if args.algorithm == 'ADEPT':
                hiddens = model(input, input_attention_mask).hidden_states
            elif args.algorithm == 'ADEPT-finetuning' or args.algorithm == 'DPCE':
                hiddens = model.bert(input).hidden_states

        return hiddens

    def attribute_vector_example():
        if args.bias == 'gender':
            d = 2
        elif args.bias == 'religion':
            d = 3
        attributes_hiddens = {f'attribute{i}': [] for i in range(d)}

        dataloaders, _, distribution = create_dataloader(args, train_datasets, tokenizer, train=True)
        for key in distribution:
            if key != 'neutral':
                inputs, labels, inputs_attention_mask, _ = next(dataloaders[key])
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
                attributes_hiddens[key].append(torch.sum(hiddens * onehot, 1) / labels.size(1))

        attribute_size = len(data['train']['example'])
        for i in range(attribute_size - 1):
            attributes_hiddens[f'attribute{i}'] = torch.mean(torch.cat(attributes_hiddens[f'attribute{i}'], 0), 0).detach().unsqueeze(0)

        return attributes_hiddens

    def forward(attributes_hiddens, dataloaders, key):
        inputs = next(dataloaders[key])
        if len(inputs) == 4:
            inputs, labels, inputs_attention_mask, labels_attention_mask = inputs
            labels = labels.to(args.device)
            labels_attention_mask = labels_attention_mask.to(args.device)
        else:
            inputs, inputs_attention_mask = inputs
            labels = None
            labels_attention_mask = None
        inputs = inputs.to(args.device)
        inputs_attention_mask = inputs_attention_mask.to(args.device)
        if args.model_type == 'roberta':
            all_layer_hiddens = model(inputs, inputs_attention_mask).hidden_states
            if 'neutral' != key:
                with torch.no_grad():
                    all_layer_original_hiddens = original_model(inputs, inputs_attention_mask).hidden_states
        elif args.model_type == 'bert':
            all_layer_hiddens = model(inputs, inputs_attention_mask).hidden_states
            if 'neutral' != key:
                with torch.no_grad():
                    all_layer_original_hiddens = original_model(inputs, inputs_attention_mask).hidden_states

        all_layer_hiddens = torch.stack(all_layer_hiddens, 2)
        if 'neutral' != key:
            all_original_hiddens =  torch.stack(all_layer_original_hiddens, 2)
            all_original_hiddens = all_original_hiddens.detach()
        if args.debias_layer == 'all':
            target_layer_hiddens = all_layer_hiddens
            target_original_hiddens = all_layer_hiddens
        else:
            if args.debias_layer == 'first':
                idx = 0
            elif args.debias_layer == 'last':
                idx = -1
            target_layer_hiddens = all_layer_hiddens[:,:,idx]
            target_layer_hiddens = target_layer_hiddens.unsqueeze(2)
            if 'neutral' != key:
                target_original_hiddens = all_original_hiddens[:,:,idx]
                target_original_hiddens = target_original_hiddens.unsqueeze(2)
            else:
                attributes_hiddens = {key: value[:,idx,:].unsqueeze(1) for key, value in attributes_hiddens.items()}

        if args.loss_target == 'sentence' or labels is None:
            attributes_hiddens = {key: value.unsqueeze(1) for key, value in attributes_hiddens.items()}
        #elif args.loss_target == 'token' and key == 'neutral':
        elif args.loss_target == 'token':
            if labels.size(1) > 1:
                onehot = torch.eye(target_layer_hiddens.size(1))
                zeros = torch.zeros(1, onehot.size(0))
                onehot = torch.cat((zeros, onehot), 0)
                onehot = onehot[labels]
                onehot = torch.sum(onehot, 1)
                onehot = onehot.view(target_layer_hiddens.size(0), -1, 1, 1)
            else:
                onehot = torch.eye(target_layer_hiddens.size(1))[labels].view(target_layer_hiddens.size(0), -1, 1, 1)
            onehot = onehot.to(args.device)
            target_layer_hiddens = torch.sum(target_layer_hiddens * onehot, 1).unsqueeze(1) / labels.size(1)
            if 'neutral' != key:
                target_original_hiddens = torch.sum(target_original_hiddens * onehot, 1).unsqueeze(1) / labels.size(1)
            else:
                if args.algorithm == 'ADEPT' or args.algorithm == 'ADEPT-finetuning':
                    attributes_hiddens = torch.cat(list(attributes_hiddens.values()), dim=0)
                elif args.algorithm == 'DPCE':
                    attributes_hiddens = {key: value.expand(target_layer_hiddens.size(0),
                                                        1,
                                                        value.size(1),
                                                        value.size(2))
                                      for key, value in attributes_hiddens.items()}
        if args.algorithm == 'ADEPT' or args.algorithm == 'ADEPT-finetuning':
            loss = 0
            if 'neutral' == key:
                relative_distance = calculate_group_to_one_relative_distance_asymmetric(target_layer_hiddens, attributes_hiddens)
                relative_distance_shape0 = relative_distance.shape[0]
                for i in range(relative_distance_shape0):
                    for j in range(i + 1, relative_distance_shape0):
                        loss += JS_divergence(relative_distance[i], relative_distance[j])
                loss /= relative_distance_shape0 * (relative_distance_shape0 - 1) / 2
                loss *= alpha
            else:
                if args.KL_divergence:
                    _all_layer_hiddens = torch.softmax(all_layer_hiddens, dim=-1)
                    _all_original_hiddens = torch.softmax(all_original_hiddens, dim=-1)
                    loss += KL_divergence(_all_layer_hiddens, _all_original_hiddens)
                    _all_layer_hiddens = torch.softmax(-all_layer_hiddens, dim=-1)
                    _all_original_hiddens = torch.softmax(-all_original_hiddens, dim=-1)
                    loss += KL_divergence(_all_layer_hiddens, _all_original_hiddens)
                else:
                    loss += criterion_ms(all_layer_hiddens, all_original_hiddens, 3)
                loss *= beta
        elif args.algorithm == 'DPCE':
            if 'neutral' == key:
                loss = 0
                for attribute_hiddens in attributes_hiddens.values():
                    tmp_loss = criterion_ip(target_layer_hiddens, attribute_hiddens)
                    tmp_loss = tmp_loss ** 2
                    tmp_loss *= alpha
                    loss += tmp_loss
            else:
                loss = criterion_ms(all_layer_hiddens, all_original_hiddens, 3)
                loss *= beta
        return loss

    def evaluate(model, attributes_hiddens, dev_dataloaders, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = args.output_dir

        if args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir, exist_ok=True)

        args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("Num examples = %d", dev_example_num)
        logger.info("Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        model.eval()
        #criterion.eval()

        for key in tqdm(dev_distribution):
            with torch.no_grad():
                loss = forward(attributes_hiddens, dev_dataloaders, key)

                eval_loss += loss.item()

                model.zero_grad()
                original_model.zero_grad()

        return eval_loss

    criterion_ms = mean_square
    criterion_ip = inner_product
    original_model.eval()

    alpha, beta = args.weighted_loss
    alpha = float(alpha)
    beta = float(beta)

    train_loss = 0.0

    for _ in train_iterator:

        random.shuffle(train_distribution)
        epoch_iterator = tqdm(train_distribution, desc="Iteration", disable=args.local_rank not in [-1, 0])

        model.eval()
        with torch.no_grad():
            attributes_hiddens = attribute_vector_example()

        for step, key in enumerate(epoch_iterator):
            model.train()

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            loss = forward(attributes_hiddens, train_dataloaders, key)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                original_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("global_step = %s, train loss = %s", global_step, train_loss)
                    train_loss = 0.0
                    # Log metrics
                    best_loss, best_step = save_best_model(best_loss, best_step, dev_dataloaders)
                    dev_dataloaders, dev_example_num, dev_distribution = create_dataloader(args, dev_datasets, tokenizer, train=False)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            train_dataloaders, train_example_num, train_distribution = create_dataloader(args, train_datasets, tokenizer, train=True)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    dev_dataloaders, dev_example_num, dev_distribution = create_dataloader(args, dev_datasets, tokenizer, train=False)
    best_loss, best_step = save_best_model(best_loss, best_step, dev_dataloaders)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

def main():
    model_args, args = get_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)

    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{args.log_dir}/{timestr}.log')],
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir, revision=model_args.model_revision)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, revision=model_args.model_revision)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    config.output_hidden_states = 'true'

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
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

    if model_args.model_name_or_path:
        if args.algorithm == 'ADEPT':
            model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
        else:
            model = AutoModelWithLMHead.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        original_model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError()
        

    # GPT-2 and GPT do not have pad.
    if tokenizer._pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        original_model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)
    original_model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    data = torch.load(args.data_file)

    attributes_examples = data['attributes_examples']
    attributes_labels = data['attributes_labels']
    neutral_examples = data['neutral_examples']

    if 'neutral_labels' in data:
        neutral_labels = data['neutral_labels']
        splited_data = split_data(attributes_examples, attributes_labels, neutral_examples, neutral_labels, args)
    else:
        splited_data = split_data(attributes_examples, attributes_labels, neutral_examples, None, args)

    datasets = load_and_cache_examples(splited_data, args, tokenizer)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    if args.local_rank == 0:
        torch.distributed.barrier()

    train(args, splited_data, datasets, model, original_model, tokenizer)


if __name__ == "__main__":
    main()
