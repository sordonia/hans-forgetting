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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

import torch
import random
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_curve, auc, roc_auc_score

import pandas as pd
logger = logging.getLogger(__name__)


class PairedDataset(torch.utils.data.Dataset):
    """Loops through a pair of datasets in parallel.
    """
    def __init__(self, data1, data2, match_length='max', sync=False):
        self.data1 = data1
        self.data2 = data2
        self.match_length = match_length
        self.sync = sync
        if self.sync:
            # indices of first dataset should match indices of second dataset
            assert len(self.data1) == len(self.data2)

    def __getitem__(self, indices):
        if self.sync:
            return (self.data1.__getitem__(indices), self.data2.__getitem__(indices))
        return (self.data1.__getitem__(indices[0]), self.data2.__getitem__(indices[1]))

    def lens(self):
        return (len(self.data1), len(self.data2))

    def __len__(self):
        if self.match_length == 'max':
            return max(len(self.data1), len(self.data2))
        elif self.match_length == 'min':
            return min(len(self.data1), len(self.data2))


class ProportionDataset(torch.utils.data.Dataset):
    """Loops through a pair of datasets in parallel.
    """
    def __init__(self, data1, data2, match_length='max', proportion=0.1):
        self.data1 = data1
        self.data2 = data2
        self.p = proportion
        self.match_length = match_length

    def __getitem__(self, indices):
        b1 = self.data1.__getitem__(indices[0])
        b2 = self.data2.__getitem__(indices[1])
        if (random.random() > self.p):
            return b1
        else:
            return b2

    def lens(self):
        return (len(self.data1), len(self.data2))

    def __len__(self):
        if self.match_length == 'max':
            return max(len(self.data1), len(self.data2))
        elif self.match_length == 'min':
            return min(len(self.data1), len(self.data2))


class RandomPairedSampler(torch.utils.data.sampler.Sampler):
    """Randomly samples from a PairedDataset.
    """
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        import random
        len1 = self.data_source.lens()[0]
        len2 = self.data_source.lens()[1]
        range_1 = list(range(len(self.data_source)))
        range_2 = list(range(len(self.data_source)))
        random.shuffle(range_1)
        random.shuffle(range_2)
        range_1 = [d % len1 for d in range_1]
        range_2 = [d % len2 for d in range_2]
        return iter(zip(range_1, range_2))

    def __len__(self):
        return len(self.data_source)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, guid_num=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.guid_num = guid_num
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 guid=None, guid_num=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        self.guid_num = guid_num


class PairInputFeatures(object):
    def __init__(self, input_ids_a, input_mask_a,
                 input_ids_b, input_mask_b,
                 label_id, guid=None, guid_num=None):
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.label_id = label_id
        self.guid = guid
        self.guid_num = guid_num



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    
    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a json value file."""
        df = pd.read_json(input_file, lines=True )
        return df.iterrows()


class HansProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "heuristics_train_set.txt")),
            "set")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "heuristics_evaluation_set.txt")),
            "set")

    def get_labels(self):
        """See base class."""
        return ["non-entailment", "entailment", "dummy"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s" % (line[7])
            text_a = line[5].strip()
            text_b = line[6].strip()
            label = line[0].strip()
            if i < 5:
                print(text_a, text_b, label)
            examples.append(
                InputExample(guid=guid, guid_num=int(i-1),
                             text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            guid_num = int(line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, guid_num=guid_num, text_a=text_a,
                             text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class MnliHardProcessor(MnliProcessor):
    """Processor for the MultiNLI hard data set. """ 

    def get_labels(self):
        """See base class."""
        return ["hidden", "dummy1", "dummy2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s" % (line[8])
            text_a = line[5]
            text_b = line[6]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "multinli_0.9_test_matched_unlabeled_hard.txt")),
            "set")


class MnliStressProcessor(MnliProcessor):
    """Processor for the MultiNLI hard data set. """ 

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s" % (line[8])
            text_a = line[5]
            text_b = line[6]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        for f in os.listdir(data_dir):
            if f.endswith(".txt") and "_matched" in f:
               dev_path = os.path.join(data_dir, f) 
        return self._create_examples(
            self._read_tsv(dev_path),
            "set")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpWangProcessor(DataProcessor):
    """Processor for the QQP data set (Wang et al split, arxiv.org/abs/1702.03814).
    Used in experiments in PAWS paper (https://github.com/google-research-datasets/paws).
    Shuffle lines of train.tsv as shuffled_train.tsv to use this task. """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "shuffled_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_qqp.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[1]
                text_b = line[2]
                label = line[0]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class QqpWangTestProcessor(QqpWangProcessor):
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


class PawsQqpProcessor(DataProcessor):
    """ PAWS-QQP task class, a subset of PAWS dataset.
	Used in experiments in PAWS paper (https://github.com/google-research-datasets/paws).
	Shuffle lines of train.tsv as shuffled_train.tsv to use this task. """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "shuffled_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_and_test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i - 1)
            try:
                text_a = line[1]
                text_b = line[2]
                label = line[3]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PawsQqpAllValProcessor(PawsQqpProcessor):
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "shuffled_train_dev_and_test.tsv")), "dev")


class PawsWikiProcessor(PawsQqpProcessor):
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class FeverProcessor(DataProcessor):
    """Processor for the FEVER data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "fever.train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "fever.dev.jsonl")), "dev_fever")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "fever_symmetric_generated.jsonl")),
            "test_symmetric")

    def get_labels(self):
        """See base class."""
        return ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (_, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            guid_num = i
            text_a = line['evidence_sentence']
            text_b = line['claim']
            if 'label' in line:
                label = line['label']
            elif 'gold_label' in line:
                label = line['gold_label']
            examples.append(
                InputExample(guid=guid, guid_num=guid_num,  
                             text_a=text_a, text_b=text_b, label=label))
        return examples


class FeverSymmetricProcessor(FeverProcessor):
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "fever_symmetric_generated.jsonl")),
            "fever_symmetric_generated")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        ids = [3, 2, 1]
        for i, (_, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            guid_num = i
            text_a = line[3]
            text_b = line[2]
            label = line[1]
            examples.append(
                InputExample(guid=guid, guid_num=guid_num, text_a=text_a, text_b=text_b, label=label))
        return examples


class FeverSymmetricV2Processor(FeverProcessor):
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "fever_symmetric_dev.jsonl")),
            "dev_fever_symmetric_v2")
        
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "fever_symmetric_test.jsonl")),
            "test_fever_symmetric_v2")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (_, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            guid_num = i
            text_a = line[2]
            text_b = line[1]
            label = line[5]
            examples.append(
                InputExample(guid=guid, guid_num=guid_num, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features_base(examples, label_list, max_seq_length,
                                      tokenizer, output_mode):
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        assert example.text_b is not None
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
        if len(tokens_b) > max_seq_length:
            tokens_b = tokens_b[:max_seq_length]
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
        input_mask_a = [1] * len(input_ids_a) + [0] * (max_seq_length - len(input_ids_a))
        input_mask_b = [1] * len(input_ids_b) + [0] * (max_seq_length - len(input_ids_b))
        input_ids_a += [0] * (max_seq_length - len(input_ids_a))
        input_ids_b += [0] * (max_seq_length - len(input_ids_b))
        assert len(input_ids_a) == max_seq_length
        assert len(input_ids_b) == max_seq_length
        assert len(input_mask_a) == max_seq_length
        assert len(input_mask_b) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens_a: %s" % " ".join([str(x) for x in tokens_a]))
            logger.info("tokens_b: %s" % " ".join([str(x) for x in tokens_b]))
            logger.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
            logger.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
            logger.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
            logger.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            PairInputFeatures(guid=example.guid,
                              guid_num=example.guid_num,
                              input_ids_a=input_ids_a,
                              input_ids_b=input_ids_b,
                              input_mask_a=input_mask_a,
                              input_mask_b=input_mask_b,
                              label_id=label_id))
    return features


def convert_examples_to_features_bert(examples, label_list, max_seq_length,
                                      tokenizer, output_mode,
                                      cls_token_at_end=False,
                                      cls_token='[CLS]',
                                      cls_token_segment_id=1,
                                      sep_token='[SEP]',
                                      sep_token_extra=False,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      sequence_a_segment_id=0, 
                                      sequence_b_segment_id=1,
                                      mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(guid=example.guid,
                          guid_num=example.guid_num,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, avg="binary"):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=avg)
    f1_perclass = f1_score(y_true=labels, y_pred=preds, average=None)
    return {
        "acc": acc,
        "f1": f1,
        "f1_class": f1_perclass,
        "acc_and_f1": round(100*(acc + f1) / 2, 1)
    }


def auc_binary_precison_recall_curve(probas_pred, labels, positive_label=None):
    """x-axis recall (monotonically increasing or decreasing), y-axis corresponing precision."""
    precision, recall, _ = precision_recall_curve(labels, probas_pred, positive_label)
    # recall: Decreasing recall values such that element i is the recall of predictions with
    # score >= thresholds[i] and the last element is 0.
    return auc(recall, precision)


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels, probs=None):
    assert len(preds) == len(labels)
    if task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    if task_name in ["fever", "fever-symmetric-r1", "fever-symmetric-r2"]:
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "aux":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-hard":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-stress":
        return acc_and_f1(preds, labels, avg="micro")
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qqp-wang":
        return {
            "acc": simple_accuracy(preds, labels), 
            "auc": auc_binary_precison_recall_curve(probs, labels, positive_label=1), 
            "roc-auc": roc_auc_score(labels, probs)
        }
    elif task_name == "paws-qqp":
        return {
            "acc": simple_accuracy(preds, labels), 
            "auc": auc_binary_precison_recall_curve(probs, labels, positive_label=1)
            , "roc-auc": roc_auc_score(labels, probs)
        }
    elif task_name == "qqp-wang-test":
        return {"acc": simple_accuracy(preds, labels),
        "auc": auc_binary_precison_recall_curve(probs, labels, positive_label=1)
        , "roc auc": roc_auc_score(labels, probs)}
    elif task_name == "paws-qqp-all-val":
        return {"acc": simple_accuracy(preds, labels),
        "auc": auc_binary_precison_recall_curve(probs, labels, positive_label=1)}
    elif task_name == "paws-wiki":
        return {"acc": simple_accuracy(preds, labels), 
        "auc": auc_binary_precison_recall_curve(probs, labels, positive_label=1)}
    else:
        raise KeyError(task_name)


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "hans": HansProcessor,
    "mnli-hard": MnliHardProcessor,
    "mnli-stress": MnliStressProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "fever": FeverProcessor,
    "fever-symmetric-r1": FeverSymmetricProcessor,
    "fever-symmetric-r2": FeverSymmetricV2Processor,
    "qqp-wang": QqpWangProcessor,
    "qqp-wang-test": QqpWangTestProcessor,
    "paws-qqp": PawsQqpProcessor,
    "paws-wiki": PawsWikiProcessor,
    "paws-qqp-all-val": PawsQqpAllValProcessor
}

output_modes = {
    "cola": "classification",
    "hans": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mnli-hard": "classification",
    "mnli-stress": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "fever": "classification",
    "fever-symmetric-r1": "classification",
    "fever-symmetric-r2": "classification",
    "qqp-wang": "classification",
    "qqp-wang-test": "classification",
    "paws-qqp": "classification",
    "paws-wiki": "classification",
    "paws-qqp-all-val": "classification"
}


GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "fever": 3,
    "fever-symmetric-r1": 3,
    "fever-symmetric-r2": 3,
    "qqp-wang": 2,
    "qqp-wang-test": 2,
    "paws-qqp": 2,
    "paws-wiki": 2,
    "paws-qqp-all-val": 2,
    "counterfactual-nli": 3
}
