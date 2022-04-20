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
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]

@dataclass(frozen=True)
class InputFusion:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    text: Optional[InputFeatures]
    amr: Optional[InputFeatures]
    label: Optional[int]



class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"



class MultipleChoiceDataset(Dataset):
    """
    PyTorch multiple choice dataset class
    """

    features: List[InputFusion]

    def __init__(
        self,
        data_args,
        tokenizer_text: PreTrainedTokenizer,
        tokenizer_amr: PreTrainedTokenizer,
        task: str,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        
        dataset_text = datasets.load_from_disk(data_args.data_set_path_text)
        dataset_amr = datasets.load_from_disk(data_args.data_set_path_amr)

        tokenizer_name = re.sub('[^a-z]+', ' ', tokenizer_text.name_or_path).title().replace(' ', '')
        cached_features_file = os.path.join(
            '.cache',
            task,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer_name,
                str(data_args.max_seq_length_text),
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
                    examples_text = dataset_text['validation']
                    examples_amr = dataset_amr['validation']
                elif mode == Split.test:
                    examples_text = dataset_text['test']
                    examples_amr = dataset_amr['test']                
                elif mode == Split.train:
                    examples_text = dataset_text['train']
                    examples_amr = dataset_amr['train']      
                
                self.features = convert_examples_to_features(
                    examples_text,
                    examples_amr,
                    data_args.max_seq_length_text,
                    data_args.max_seq_length_amr,
                    tokenizer_text,
                    tokenizer_amr,
                )
                logger.info("Training examples: %s", len(self.features))

                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFusion:
        return self.features[i]


def convert_examples_to_features(
    examples_text: datasets.Dataset,
    examples_amr: datasets.Dataset,
    max_length_text: int,
    max_length_amr: int,
    tokenizer_text: PreTrainedTokenizer,
    tokenizer_amr: PreTrainedTokenizer
) -> List[InputFusion]:
    """
    Loads a data file into a list of `InputFusion`
    """
    features = []
    for (ex_index, _) in tqdm.tqdm(enumerate(examples_text), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples_text)))

        input_text = get_input_feature_text(examples_text[ex_index], tokenizer_text, max_length_text)
        input_amr = get_input_feature_amr(examples_amr[ex_index], tokenizer_amr, max_length_amr)

        label = examples_text[ex_index]['label']

        features.append(
            InputFusion(
                text=input_text,
                amr=input_amr,
                label=label
            )
        )

    return features    

def get_input_feature_amr(example, tokenizer_amr, max_length):
    choices_inputs = []
    for idx, ending in enumerate(example['endings']):
        context = example['context']
        question = example['question']

        inputs = [tokenizer_amr.bos_token] \
            + context.split() \
            + [tokenizer_amr.eos_token] \
            + [tokenizer_amr.eos_token] \
            + question.split() \
            + ending.split() \
            + [tokenizer_amr.eos_token] \

        if len(inputs) > max_length:
            logger.error("Input too long: implementation does not support truncate.")

        choices_inputs.append(inputs)

    input_ids = amr_batch_encode(tokenizer_amr, choices_inputs, max_length = max_length, pad_to_max_length=False)

    model_inputs = {}
    model_inputs['input_ids'] = input_ids

    input_features = tokenizer_amr.pad(
        model_inputs,
        padding='max_length',
        max_length=max_length,
    )

    feature = InputFeatures(
        input_ids=input_features['input_ids'],
        attention_mask=input_features['attention_mask'],
        token_type_ids=None,
    )
    return feature
    

def get_input_feature_text(example, tokenizer_text, max_length):
    choices_inputs = []

    for idx, ending in enumerate(example['endings']):
        context = example['context']
        question = example['question']
        inputs = tokenizer_text(
            context,
            question + ending ,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        choices_inputs.append(inputs)
    
    input_ids = [x["input_ids"] for x in choices_inputs]
    attention_mask = (
        [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
    )
    token_type_ids = (
        [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
    )

    feature = InputFeatures(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    return feature

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