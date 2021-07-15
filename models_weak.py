# StackedSelfAttentionEncoder
import torch
import collections
import os
import pickle
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import logging
import copy
import math
import json
import numpy
from pytorch_transformers import (
    tokenization_bert, tokenization_utils, configuration_utils,
    PreTrainedModel, PretrainedConfig)
from pytorch_transformers.file_utils import cached_path, WEIGHTS_NAME, TF_WEIGHTS_NAME


logger = logging.getLogger(__name__)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    id_to_tokens = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
        id_to_tokens[index] = token
    return vocab, id_to_tokens


def load_txt_embeddings(path):
    import pandas as pd
    import csv
    words = pd.read_csv(path, sep=" ", index_col=0,
                        na_values=None, keep_default_na=False, header=None,
                        encoding="utf-8", quoting=csv.QUOTE_NONE)
    matrix = words.values
    index_to_word = list(words.index)
    word_to_index = {
        word: ind for ind, word in enumerate(index_to_word)
    }
    print("Loaded", len(index_to_word), "embeddings")
    return matrix, index_to_word, word_to_index


def extract_subset_from_glove(glove_path, dictionary, output_dir):
    import pandas as pd
    import numpy as np
    import pickle
    vocab, index_to_word = load_vocab(dictionary)
    print("Filtering", len(vocab), "embeddings.")
    matrix, _, word_to_index = load_txt_embeddings(glove_path)
    unk_word = matrix.mean(0)
    subset_matrix = np.zeros((len(vocab), matrix.shape[1])) + unk_word[None, :]
    num_unks = 0
    for index, token in index_to_word.items():
        ind = word_to_index.get(token, -1)
        if ind > -1:
            subset_matrix[index] = matrix[ind]
        else:
            num_unks += 1

    print("Filtering done, num unks", num_unks)
    with open(output_dir + "/embeddings.pkl", "wb") as f:
        pickle.dump(dict(word_to_index=vocab, embeddings=subset_matrix), f)


def load_embeddings(path):
    resource = pickle.load(open(path, 'rb'))
    word_to_index = resource['word_to_index']
    matrix = resource['embeddings']
    index_to_word = [(i, w) for w, i in word_to_index.items()]
    return matrix, word_to_index, index_to_word


class BaselineTokenizer(tokenization_utils.PreTrainedTokenizer):
    @classmethod
    def from_pretrained(cls, model_name_or_path, do_lower_case=False, **kwargs):
        vocab_file = kwargs.pop('vocab_file', None)
        # Load vocab from the checkpoint
        if vocab_file is None:
            assert os.path.exists(model_name_or_path)
            vocab_file = os.path.join(model_name_or_path, "vocab.txt")
        if do_lower_case:
            logger.info("Lower casing is set to True, sure you're doing the right thing?")
        return cls(vocab_file, do_lower_case=do_lower_case)

    def __init__(self, vocab_file, do_lower_case,
                 never_split=None, tokenize_chinese_chars=True):
        super().__init__(max_len=128, vocab_file=vocab_file,
                         unk_token="[UNK]", sep_token="[SEP]",
                         pad_token="[PAD]", cls_token="[CLS]",
                         mask_token="[MASK]",
                         do_lower_case=do_lower_case,
                         never_split=never_split,
                         tokenize_chinese_chars=tokenize_chinese_chars)
        self.tokenizer = tokenization_bert.BasicTokenizer(
            do_lower_case=do_lower_case, never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars)
        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)
        logger.info("Vocabulary size: %d", len(self.vocab))

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def _tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, "vocab.txt")
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return (vocab_file,)


class BaselineConfig(PretrainedConfig):
    def __init__(self, vocab_size_or_config_json_file, **kwargs):
        super().__init__(vocab_size_or_config_json_file=vocab_size_or_config_json_file, **kwargs)
        for key, value in kwargs.items():
            self.__dict__[key] = value


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def masked_softmax(
    vector: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    memory_efficient: bool = False,
    mask_fill_value: float = -1e32) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result



class BaselineModel(PreTrainedModel):
    def save_pretrained(self, save_directory):
        """Save a model and its configuration file to a directory, so that it
           can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Save class name
        with open(os.path.join(save_directory, "class.txt"), "w") as cf:
            cf.write(self.base_model_prefix + "\n")

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        """Model_name_or_path is the type of model to be loaded.
        """
        BASELINE_MODELS_MAP = {
            "lstm-att": LSTMAtt,
            "bow": BOW,
            "bilstm": BaseBiLSTM
        }

        weights_file = None
        init_class = BASELINE_MODELS_MAP.get(model_name_or_path)
        config = kwargs.pop("config", None)

        if init_class is not None:
            assert config is not None, "A config is required when initializing a model from scratch."
            vocab = load_vocab(config.vocab_file)[0]
            model = init_class(config, vocab)
        elif os.path.isdir(model_name_or_path):
            with open(os.path.join(model_name_or_path, "class.txt"), "r") as cf:
                init_class = BASELINE_MODELS_MAP.get(cf.readlines()[0].rstrip('\n'))
            # Load config file
            if config is None:
                config = BaselineConfig.from_pretrained(model_name_or_path)
            # Load vocab
            vocab, _ = load_vocab(os.path.join(model_name_or_path, "vocab.txt"))
            model = init_class(config, vocab)
            # Load weights
            weights_file = os.path.join(model_name_or_path, WEIGHTS_NAME)
            if weights_file is not None:
                # Load from a PyTorch state_dict
                state_dict = torch.load(weights_file, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info("Loaded pretrained model.")

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()
        return model


class BOW(BaselineModel):
    config_class = BaselineConfig
    pretrained_model_archive_map = None
    load_tf_weights = None
    base_model_prefix = "bow"

    def __init__(self, config, vocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.dropout = torch.nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(
            len(self.vocab), config.embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(config.embedding_dim, 512),
            nn.Tanh())
        self.classifier = nn.Sequential(
            nn.Linear(4 * 512, 200),
            nn.Tanh(),
            nn.LayerNorm(200),
            self.dropout,
            nn.Linear(200, 3),
        )
        self.init_weights()

    def init_weights(self):
        ext_embeddings, ext_word_to_index, _ = load_embeddings(
            self.config.embedding_file)
        embeddings = self.embedding.weight.data.cpu().numpy()
        word_found = 0
        for word, index in self.vocab.items():
            if word in ext_word_to_index:
                embeddings[index] = ext_embeddings[ext_word_to_index[word]]
                word_found += 1
        logger.info('Embeddings found %d / %d', word_found, len(self.vocab))
        embeddings = torch.from_numpy(embeddings).to(self.embedding.weight.device)
        self.embedding.load_state_dict({'weight': embeddings})
        self.embedding.weight.requires_grad = True

    def forward(self, input_ids_a, input_ids_b, input_mask_a=None, input_mask_b=None, labels=None, reduction='mean'):
        s1 = input_ids_a
        s2 = input_ids_b
        s1_mask = input_mask_a
        s2_mask = input_mask_b
        if s1_mask is None:
            s1_mask = torch.ones(s1.size(0), s1.size(1)).to(s1.device)
        if s2_mask is None:
            s2_mask = torch.ones(s2.size(0), s2.size(1)).to(s2.device)
        # Similarity matrix
        s1 = (self.proj(self.embedding(s1)) * s1_mask.unsqueeze(-1)).sum(1) / s1_mask.sum(-1, keepdim=True)
        s2 = (self.proj(self.embedding(s2)) * s2_mask.unsqueeze(-1)).sum(1) / s2_mask.sum(-1, keepdim=True)
        h = torch.cat((s1 * s2, torch.abs_(s1 - s2), s1, s2), 1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = nn.CrossEntropyLoss(reduction=reduction)(logits, labels)
            outputs = (loss,) + outputs
        return outputs


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pooling='none'):
        super(BiLSTM, self).__init__()
        self.pooling = pooling
        self.cell = nn.GRU(
            embedding_dim, hidden_dim,
            bidirectional=True, batch_first=True)


    def forward(self, embeddings, mask, **kwargs):
        seq_lengths = mask.sum(1)
        max_length = embeddings.size(1)
        seq_lengths_sort, idx_sort = seq_lengths.sort(0, descending=True)
        embeddings_sort = embeddings.index_select(0, idx_sort)
        seq_lengths_sort = seq_lengths_sort.cpu().numpy().astype('int32')
        packed_input = pack_padded_sequence(embeddings_sort, seq_lengths_sort, batch_first=True)
        _, idx_unsort = idx_sort.sort(0)
        self.cell.flatten_parameters()
        all_states, source_hn = self.cell(packed_input)
        if type(source_hn) == tuple:
            source_hn = source_hn[0]
        all_states, _ = pad_packed_sequence(all_states, batch_first=True)
        all_states = all_states.index_select(0, idx_unsort)
        act_length = all_states.size(1)
        if max_length - act_length > 0:
            all_states = F.pad(all_states, (0, 0, 0, int(max_length - act_length)), "constant")
        if self.pooling == 'max':
            all_states.masked_fill_(~mask[:, :, None].bool(), -10000)
            pooled, _ = torch.max(all_states, 1)
            all_states.masked_fill_(~mask[:, :, None].bool(), 0)
        elif self.pooling == 'mean':
            assert False
            all_states.masked_fill_(~mask[:, :, None], 0)
            pooled = torch.sum(all_states, 1) / mask.float().sum(1)[:, None]
        else:
            pooled = torch.cat((source_hn[0], source_hn[1]), 1)
            pooled = pooled.index_select(0, idx_unsort)
        return pooled, all_states


class LSTMAtt(BaselineModel):
    config_class = BaselineConfig
    pretrained_model_archive_map = None
    load_tf_weights = None
    base_model_prefix = "lstm-att"

    def __init__(self, config, vocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.dropout = torch.nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(
            len(self.vocab),
            config.embedding_dim)
        self.modeling_layer = BiLSTM(2 * 512, config.hidden_dim // 2,
                                     pooling='max')
        self.proj = nn.Sequential(
            nn.Linear(config.embedding_dim, 512),
            nn.Tanh())
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, 200),
            nn.Tanh(),
            nn.LayerNorm(200),
            self.dropout,
            nn.Linear(200, 3),
        )
        self.init_weights()

    def init_weights(self):
        ext_embeddings, ext_word_to_index, _ = load_embeddings(
            self.config.embedding_file)
        embeddings = self.embedding.weight.data.cpu().numpy()
        word_found = 0
        for word, index in self.vocab.items():
            if word in ext_word_to_index:
                embeddings[index] = ext_embeddings[ext_word_to_index[word]]
                word_found += 1
        logger.info('Embeddings found %d / %d', word_found, len(self.vocab))
        embeddings = torch.from_numpy(embeddings).to(self.embedding.weight.device)
        self.embedding.load_state_dict({'weight': embeddings})
        self.embedding.weight.requires_grad = True

    def forward(self, input_ids_a, input_ids_b, input_mask_a=None, input_mask_b=None, labels=None, reduction='mean'):
        s1 = input_ids_a
        s2 = input_ids_b
        s1_mask = input_mask_a
        s2_mask = input_mask_b
        if s1_mask is None:
            s1_mask = torch.ones(s1.size(0), s1.size(1)).to(s1.device)
        if s2_mask is None:
            s2_mask = torch.ones(s2.size(0), s2.size(1)).to(s2.device)
        # Similarity matrix
        s1 = self.proj(self.embedding(s1))
        s2 = self.proj(self.embedding(s2))
        # Similarity matrix
        # Shape: (batch_size, s2_length, s1_length)
        similarity_mat = torch.matmul(s2, s1.permute(0, 2, 1))
        # s2 representation
        # Shape: (batch_size, s2_length, s1_length)
        s2_s1_attn = masked_softmax(similarity_mat, s1_mask)
        # Shape: (batch_size, s2_length, encoding_dim)
        s2_s1_vectors = weighted_sum(s1, s2_s1_attn)
        # batch_size, seq_len, 4*enc_dim
        s2_w_context = torch.cat([s2, s2_s1_vectors], 2)
        # s1 representation, using same attn method as for the s2
        # representation
        s1_s2_attn = masked_softmax(similarity_mat.transpose(1, 2).contiguous(), s2_mask)
        # Shape: (batch_size, s1_length, encoding_dim)
        s1_s2_vectors = weighted_sum(s2, s1_s2_attn)
        s1_w_context = torch.cat([s1, s1_s2_vectors], 2)
        s1 = self.dropout(self.modeling_layer(s1_w_context, s1_mask)[0])
        s2 = self.dropout(self.modeling_layer(s2_w_context, s2_mask)[0])
        h = torch.cat((s1 * s2, torch.abs_(s1 - s2), s1, s2), 1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs


class BaseBiLSTM(BaselineModel):
    config_class = BaselineConfig
    pretrained_model_archive_map = None
    load_tf_weights = None
    base_model_prefix = "bilstm"

    def __init__(self, config, vocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.dropout = torch.nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(
            len(self.vocab),
            config.embedding_dim)
        self.modeling_layer = BiLSTM(512, config.hidden_dim // 2,
                                     pooling='max')
        self.proj = nn.Sequential(
            nn.Linear(config.embedding_dim, 512),
            nn.Tanh())
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, 200),
            nn.Tanh(),
            nn.LayerNorm(200),
            self.dropout,
            nn.Linear(200, 3),
        )
        self.init_weights()

    def init_weights(self):
        ext_embeddings, ext_word_to_index, _ = load_embeddings(
            self.config.embedding_file)
        embeddings = self.embedding.weight.data.cpu().numpy()
        word_found = 0
        for word, index in self.vocab.items():
            if word in ext_word_to_index:
                embeddings[index] = ext_embeddings[ext_word_to_index[word]]
                word_found += 1
        logger.info('Embeddings found %d / %d', word_found, len(self.vocab))
        embeddings = torch.from_numpy(embeddings).to(self.embedding.weight.device)
        self.embedding.load_state_dict({'weight': embeddings})
        self.embedding.weight.requires_grad = True

    def forward(self, input_ids_a, input_ids_b, input_mask_a=None, input_mask_b=None, labels=None, reduction='mean'):
        s1 = input_ids_a
        s2 = input_ids_b
        s1_mask = input_mask_a
        s2_mask = input_mask_b
        if s1_mask is None:
            s1_mask = torch.ones(s1.size(0), s1.size(1)).to(s1.device)
        if s2_mask is None:
            s2_mask = torch.ones(s2.size(0), s2.size(1)).to(s2.device)
        # Similarity matrix
        s1 = self.proj(self.embedding(s1))
        s2 = self.proj(self.embedding(s2))
        s1 = self.dropout(self.modeling_layer(s1, s1_mask)[0])
        s2 = self.dropout(self.modeling_layer(s2, s2_mask)[0])
        h = torch.cat((s1 * s2, torch.abs_(s1 - s2), s1, s2), 1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss,) + outputs
        return outputs
