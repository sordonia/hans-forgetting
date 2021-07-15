import torch
from utils_glue import (processors, output_modes, convert_examples_to_features_base,
                        convert_examples_to_features_bert)
from pathlib import Path
from torch.utils.data import TensorDataset, Dataset
import logging
import numpy as np


logger = logging.getLogger(__name__)


def load_and_cache_examples(
        data_dir,
        model_name_or_path,
        model_type,
        max_seq_length,
        task,
        tokenizer,
        evaluate=False,
        test=False,
    ):
    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from cache or dataset file
    cached_features_file = \
        Path(data_dir) / 'cached_{}_{}_{}_{}'.format(
            'test' if test else 'dev' if evaluate else 'train',
            list(filter(None, model_name_or_path.split('/'))).pop(),
            str(max_seq_length),
            str(task))

    if cached_features_file.exists():
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and model_type in ['roberta']:
            # HACK (label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if evaluate and test:
            examples = processor.get_test_examples(data_dir)
        elif evaluate:
            examples = processor.get_dev_examples(data_dir)
        else:
            examples = processor.get_train_examples(data_dir)

        if model_type == 'baseline':
            features = convert_examples_to_features_base(
                examples, label_list, max_seq_length, tokenizer, output_mode,
            )
        else:
            features = convert_examples_to_features_bert(
                examples, label_list, max_seq_length, tokenizer, output_mode,
                cls_token_at_end=bool(model_type in ['xlnet']),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(model_type in ['roberta']),
                pad_on_left=bool(model_type in ['xlnet']),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
            )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    def _create_dataset(features):
        if model_type in ['baseline']:
            ia = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
            ib = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
            ima = torch.tensor([f.input_mask_a for f in features], dtype=torch.long)
            imb = torch.tensor([f.input_mask_b for f in features], dtype=torch.long)
            all_guids = torch.tensor([f.guid_num if f.guid_num else 0 for f in features],
                                     dtype=torch.long)
            if output_mode == "classification":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            elif output_mode == "regression":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
            dataset = TensorDataset(ia, ib, ima, imb, all_label_ids, all_guids)
            return dataset
        else:
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_guids = torch.tensor([f.guid_num if f.guid_num else 0 for f in features], dtype=torch.long)
            if output_mode == "classification":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            elif output_mode == "regression":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
            dataset = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_guids)
            return dataset

    dataset = _create_dataset(features)
    logger.info("Dataset length: %s." % len(dataset))
    return features, dataset


class TensorDatasetFilter(Dataset):
    def __init__(self, tensor_dataset, examples_ids):
        self._data = tensor_dataset
        self._ids = examples_ids if (examples_ids is not None and len(examples_ids)) else np.arange(len(self._data))
        self._ids_to_row = {}
        # map from guid to row
        for num_row, example in enumerate(self._data):
            example_id = example[-1]
            self._ids_to_row[example_id.item()] = num_row

    def __len__(self):
        return self._ids.shape[0]

    def __getitem__(self, idx):
        example_id = self._ids[idx]
        example_row = self._ids_to_row[example_id]
        return self._data[example_row]
