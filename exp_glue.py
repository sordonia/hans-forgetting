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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""


from __future__ import absolute_import, division, print_function


import pickle
import argparse
import glob
from pathlib import Path
import logging
import scipy
import os
import random
import numpy as np
import torch
import pandas as pd
from pandas import json_normalize
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler,
    TensorDataset)
from models_weak import *
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils_data import TensorDatasetFilter, load_and_cache_examples
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics,
                        auc_binary_precison_recall_curve,
                        f1_score,
                        ProportionDataset,
                        convert_examples_to_features_base,
                        convert_examples_to_features_bert,
                        output_modes, processors, RandomPairedSampler, PairedDataset)


logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'baseline': (BaselineConfig, BaselineModel, BaselineTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_pandas(results):
        """Assumes results is a nested dict with {'task': {'metric': value}}
        """
        return pd.DataFrame(json_normalize(results))

        
def train(args, train_dataset, model, tokenizer, eval_dataset=None, eval_features=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir, flush_secs=60)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs
    args.warmup_steps = args.warmup_proportion * t_total

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, betas=(args.adam_beta0, 0.999), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total if args.decay_learning_rate else 1e10)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    # Shall we compute forgetting for this run
    compute_forgetting = \
        args.model_type in ['bert', 'xlnet', 'baseline'] and \
        (args.hard_examples is None and not args.training_examples_ids)

    if compute_forgetting:
        shape = (len(train_dataset), int(args.num_train_epochs))
        example_stats = dict(
            accuracy=np.zeros(shape) - 1.,
            loss=np.zeros(shape) - 1.,
            margin=np.zeros(shape) - 1.,
            probs=np.zeros(shape) - 1.)

        print('Initializing forgetting', shape)
        with open(Path(args.output_dir) / 'example_stats.pkl', 'wb') as f:
            pickle.dump(example_stats, f)

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), mininterval=10,
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    def do_eval_and_save(output_dir):
        ## initial eval
        eval_results = evaluate(
            args, model, tokenizer,
            stress_subtask=args.stress_subtask,
            eval_task_names=args.eval_tasks,
            eval_output_dir=output_dir,
            aux_dataset=eval_dataset, aux_features=eval_features)

        # Save model checkpoint at the end of the epoch
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, Path(output_dir) / 'training_args.bin')
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)
        return eval_results

    results_init = do_eval_and_save(Path(args.output_dir) / f'checkpoint-epoch--1')
    all_results = [results_init]
    print(to_pandas(all_results))

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", mininterval=10,
                              disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            model.train()

            #######################################
            # Standard BERT / Baseline finetuning #
            #######################################
            batch = tuple(t.to(args.device) for t in batch)
            if args.model_type in ['baseline']:
                inputs = {
                    'input_ids_a':      batch[0],
                    'input_ids_b':      batch[1],
                    'input_mask_a':     batch[2],
                    'input_mask_b':     batch[3],
                    'labels':           batch[4]
                }
            else:
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                    'labels':         batch[3]
                }

            outputs = model(**inputs, reduction='mean')
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss.float(), optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            accuracy = (outputs[1].max(1)[1].detach().cpu() == inputs['labels'].detach().cpu()).float().mean()
            tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            tb_writer.add_scalar('loss', loss.item(), global_step)
            tb_writer.add_scalar('acc', accuracy.item(), global_step)
            epoch_iterator.set_description_str(
                'Loss: %.3f, Acc: %.3f' % (loss.item(), accuracy.item()),
                refresh=False)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                ######################
                # Compute forgetting #
                ######################
                if compute_forgetting:
                    labels = batch[-2].cpu()
                    logits = outputs[1].cpu()
                    acc = (torch.max(logits, 1)[1] == labels)
                    log_probs = F.log_softmax(logits, 1)

                    for j, guid in enumerate(batch[-1].cpu()):
                        example_nll = -log_probs[j, labels[j].item()]  # output for correct class
                        class_prob = torch.exp(log_probs[j, labels[j].item()])  # output for correct class
                        output_correct_class = logits[j, labels[j].item()]  # output for correct class
                        sorted_output, _ = torch.sort(logits.data[j, :])
                        if acc[j]:
                            # Example classified correctly, highest incorrect class is 2nd largest output
                            output_highest_incorrect_class = sorted_output[-2]
                        else:
                            # Example misclassified, highest incorrect class is max output
                            output_highest_incorrect_class = sorted_output[-1]
                        margin = output_correct_class.item() - output_highest_incorrect_class.item()
                        # Add the statistics of the current training example to dictionary
                        assert example_stats["accuracy"][guid, epoch] == -1
                        assert example_stats["loss"][guid, epoch] == -1
                        assert example_stats["margin"][guid, epoch] == -1
                        assert example_stats["probs"][guid, epoch] == -1
                        example_stats["accuracy"][guid, epoch] = acc[j].sum().item()
                        example_stats["margin"][guid, epoch] = margin
                        example_stats["loss"][guid, epoch] = example_nll.item()
                        example_stats["probs"][guid, epoch] = class_prob.item()

                ##################
                # End forgetting #
                ##################

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, stress_subtask=args.stress_subtask, aux_dataset=eval_dataset, aux_features=eval_features)
                        for key, value in results.items():
                            tb_writer.add_scalar('dev/{}'.format(key), value, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    pass
                    
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        # dump forgetting if any
        if compute_forgetting:
            with open(Path(args.output_dir) / 'example_stats.pkl', 'wb') as f:
                pickle.dump(example_stats, f)

        output_dir = Path(args.output_dir) / f'checkpoint-epoch-{epoch}'
        if epoch == int(args.num_train_epochs) - 1:
            output_dir = Path(args.output_dir) / f'checkpoint-last'

        eval_results = do_eval_and_save(output_dir)
        for task, task_results in eval_results.items():
            logger.info(f"***** Eval results {task} *****")
            if "stress" in task:
                output_eval_file = \
                    Path(output_dir) / f"eval_results_{task}-{args.stress_subtask}.txt"
            else:
                output_eval_file = \
                    Path(output_dir) / f"eval_results_{task}.txt"
            writer = open(output_eval_file, "w")
            for key, value in task_results.items():
                logger.info("%s/%s = %s", task, key, str(value))
                writer.write("%s = %s\n" % (key, str(value)))
                tb_writer.add_scalar('dev/{}/{}'.format(task, key), value, global_step)

        all_results.append(eval_results)
        results = to_pandas(all_results)
        results.to_csv(Path(args.output_dir) / 'all_results.csv')
        print(results)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(
        args, model, tokenizer,
        stress_subtask="",
        eval_task_names=None, eval_output_dir=None,
        aux_features=None, aux_dataset=None
    ):

    if not eval_task_names:
        eval_task_names = ("mnli",) if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = tuple(
        args.output_dir if eval_output_dir is None else eval_output_dir
        for _ in range(len(eval_task_names)))

    # this tests on an optional auxiliary dataset of our choice
    if aux_features is not None:
        eval_task_names = eval_task_names + ("aux",)
        eval_outputs_dirs = eval_outputs_dirs + (eval_outputs_dirs[0],)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if eval_task == "aux":
            eval_features = aux_features
            eval_dataset = aux_dataset
        else:
            eval_features, eval_dataset = \
                load_and_cache_examples(
                    args.data_dir, args.model_name_or_path, args.model_type,
                    args.max_seq_length, eval_task, tokenizer, evaluate=True,
                    test=args.test)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(eval_task))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating", mininterval=60):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                batch = tuple(t.to(args.device) for t in batch)
                if args.model_type in ['baseline']:
                    inputs = {
                        'input_ids_a':      batch[0],
                        'input_ids_b':      batch[1],
                        'input_mask_a':     batch[2],
                        'input_mask_b':     batch[3],
                        'labels':           batch[4]
                    }
                else:
                    inputs = {
                        'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                        'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        
        logits = preds.copy()
        probas_pred = scipy.special.softmax(logits, axis=1)[:, 1]

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        if eval_task == "hans":
            preds[preds == 2] = 0

        result = compute_metrics(eval_task, preds, out_label_ids, probs=probas_pred)
        results[eval_task] = dict()
        results[eval_task].update(result)

        if eval_task in ["qqp-wang", "paws-qqp", "paws-wiki", "paws-qqp-all-val", "qqp-wang-test"]:
            if args.output_mode == "classification":
                # TODO: probabilities should be calculated exactly as is done in the model.
                auc_pr = auc_binary_precison_recall_curve(probas_pred, out_label_ids, positive_label=1)
                logger.info(f"  AUC = {auc_pr}\t(specific to {eval_task})")
                calculated_f1 = f1_score(out_label_ids, preds, average=None)
                logger.info(f"  0f1 = {calculated_f1[0]}")
                logger.info(f"  1f1 = {calculated_f1[1]}")

        if eval_task == "hans":
            output_pred_file = Path(eval_output_dir) / "hans_preds.txt"
            with open(output_pred_file, "w") as f:
                for i, (pred, out_label) in enumerate(zip(preds, out_label_ids)):
                    f.write("ex%d,%s\n" % (i, "entailment" if pred == 1 else "non-entailment"))

        if eval_task == "mnli-hard":
            output_pred_file = Path(eval_output_dir) / "hard_preds.txt"
            with open(output_pred_file, "w") as f:
                f.write('pairID,gold_label\n')
                for i, (pred, out_label) in enumerate(zip(preds, out_label_ids)):
                    f.write("%s,%s\n" % (eval_features[i].guid, "entailment" if pred == 1 else "contradiction" if pred == 0 else "neutral"))

    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--training_examples_ids", default=None, type=str)
    parser.add_argument("--hard_examples", default=None, type=str)
    parser.add_argument("--hard_type", default=None, type=str)
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--do_lower_case", type=str,
                        help="Set this flag if you are using an uncased model.",
                        required=True)
    ## Loading options
    parser.add_argument("--avg_models", type=str, required=False,
                        help="Path to avg model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--load_model", type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    ##
    parser.add_argument("--proportion", default=0., type=float)
    parser.add_argument("--adam_beta0", default=0.9, type=float)
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--stress_subtask", default=None, type=str, required=False,
                        help="The name of the stress test")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--decay_learning_rate", default="True", type=str)
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0., type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--eval_tasks", default=['mnli', 'hans'], nargs='+', required=False,
                        help="The name of the tasks to evaluate during training selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--test", action='store_true',
                        help="use test set to evaluate")
    args = parser.parse_args()

    args.do_lower_case = eval(args.do_lower_case)
    args.decay_learning_rate = eval(args.decay_learning_rate)

    if args.output_dir and \
            os.path.exists(args.output_dir) and \
            os.listdir(args.output_dir) and \
            args.do_train and \
            not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists. Use --overwrite_output_dir.".format(
            args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    set_seed(args)

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.load_model:
        # Load model from checkpoint here
        config = config_class.from_pretrained(
            args.load_model, num_labels=num_labels, finetuning_task=args.task_name)
        tokenizer = tokenizer_class.from_pretrained(args.load_model, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.load_model, from_tf=False, config=config)
    else:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels, finetuning_task=args.task_name)
        tokenizer_kwargs = dict(vocab_file=config.vocab_file) if args.model_type == 'baseline' else dict()
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case,
            **tokenizer_kwargs)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=False, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Prepare training
    if args.do_train:
        _, train_dataset = load_and_cache_examples(args.data_dir, args.model_name_or_path, args.model_type,
                                                   args.max_seq_length, args.task_name, tokenizer, evaluate=False)
        hard_examples_ids = None
        if args.hard_examples:
            # filter training dataset with hard examples ids
            with open(args.hard_examples, 'rb') as f:
                hard_examples_stats = pickle.load(f)
                hard_examples_ids = hard_examples_stats[args.hard_type]
        elif args.training_examples_ids:
            # other format of training examples ids
            with open(args.training_examples_ids, 'r') as f:
                hard_examples_ids = np.array(f.read().strip().split(','), dtype='int64')

        train_dataset = TensorDatasetFilter(train_dataset, hard_examples_ids)
        if hard_examples_ids is not None:
            logger.info("Filtering dataset using hard examples: %s", len(train_dataset))
        train(args, train_dataset, model, tokenizer)
    else:
        eval_results = evaluate(
            args, model, tokenizer,
            stress_subtask=args.stress_subtask,
            eval_task_names=args.eval_tasks,
            eval_output_dir=args.load_model,)
        all_results = [eval_results]
        print(to_pandas(all_results))
    return


if __name__ == "__main__":
    main()
