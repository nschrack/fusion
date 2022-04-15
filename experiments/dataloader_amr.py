import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm
import re

from filelock import FileLock
from transformers import PreTrainedTokenizer
import datasets

import torch
from torch.utils.data.dataset import Dataset


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"



class MultipleChoiceDataset(Dataset):
    """
    PyTorch multiple choice dataset class
    """

    features: List[InputFeatures]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        dataset = datasets.load_dataset('lex_glue', task)
        tokenizer_name = re.sub('[^a-z]+', ' ', tokenizer.name_or_path).title().replace(' ', '')
        cached_features_file = os.path.join(
            '.cache',
            task,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer_name,
                str(max_seq_length),
                task,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        if not os.path.exists(os.path.join('.cache', task)):
            if not os.path.exists('.cache'):
                os.mkdir('.cache')
            os.mkdir(os.path.join('.cache', task))
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {task}")
                if mode == Split.dev:
                    examples = dataset['validation']
                elif mode == Split.test:
                    examples = dataset['test']
                elif mode == Split.train:
                    examples = dataset['train']
                logger.info("Training examples: %s", len(examples))
                self.features = convert_examples_to_features(
                    examples,
                    max_seq_length,
                    tokenizer,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def convert_examples_to_features(
    examples: datasets.Dataset,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, ending in enumerate(example['endings']):
            context = example['context']

            inputs = [tokenizer.bos_token] \
                + context.split() \
                + [tokenizer.eos_token] \
                + [tokenizer.eos_token] \
                + ending.split() \
                + [tokenizer.eos_token]

            if len(inputs) > max_length:
                logger.error("Input too long: implementation does not support truncate.")

            choices_inputs.append(inputs)
        input_ids = amr_batch_encode(tokenizer, choices_inputs, max_length = max_length, pad_to_max_length=True)

        model_inputs = {}
        model_inputs['input_ids'] = input_ids

        input_features = tokenizer.pad(
            model_inputs,
            padding='max_length',
            max_length=max_length,
        )

        label = example['label']

        features.append(
            InputFeatures(
                input_ids=input_features['input_ids'],
                attention_mask=input_features['attention_mask'],
                token_type_ids=None,
                label=label,
            )
        )
    return features    

def amr_batch_encode(tokenizer, input_lst, max_length = 0, pad_to_max_length=False):
    res = []
    for itm_lst in input_lst:
        res.append(
            get_ids(tokenizer, itm_lst, max_length, pad_to_max_length=pad_to_max_length)
        )

    return res

def get_ids(tokenizer, tokens, max_length=0, pad_to_max_length=False):
    token_ids = [tokenizer.encoder.get(b, tokenizer.unk_token_id) for b in tokens]
    if pad_to_max_length:
        assert max_length > 0, "Invalid max-length: {}".format(max_length)
        pad_ids = [tokenizer.pad_token_id for _ in range(max_length)]
        len_tok = len(token_ids)
        if max_length > len_tok:
            pad_ids[:len_tok] = map(int, token_ids)
        else:
            pad_ids = token_ids[:max_length]
        return pad_ids
    return token_ids