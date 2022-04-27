import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

import tqdm
import re

from filelock import FileLock
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union
from transformers.file_utils import PaddingStrategy
import datasets
import torch
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class MultipleChoiceDataset(Dataset):
    """
    PyTorch multiple choice dataset class
    """

    features: List[Dict]

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

    def __getitem__(self, i) -> Dict:
        return self.features[i]


def convert_examples_to_features(
    examples_text: datasets.Dataset,
    examples_amr: datasets.Dataset,
    max_length_text: int,
    max_length_amr: int,
    tokenizer_text: PreTrainedTokenizer,
    tokenizer_amr: PreTrainedTokenizer
) -> List[Dict]:
    """
    Loads a data file into a list of `InputFusion`
    """
    features = []
    for (ex_index, _) in tqdm.tqdm(enumerate(examples_text), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples_text)))

        text_input_ids, text_attention_mask, text_token_type_ids  = get_input_feature_text(examples_text[ex_index], tokenizer_text, max_length_text)
        amr_input_ids, amr_attention_mask = get_input_feature_amr(examples_amr[ex_index], tokenizer_amr, max_length_amr)

        label = examples_text[ex_index]['label']

        features.append({
                'text_input_ids': text_input_ids,
                'text_attention_mask': text_attention_mask,
                'text_token_type_ids': text_token_type_ids,
                'amr_input_ids': amr_input_ids,
                'amr_attention_mask': amr_attention_mask,
                'label': label
            }
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

    input_ids = amr_batch_encode(tokenizer_amr, choices_inputs)

    attention_mask = []
    for i in input_ids:
        attention_mask.append([1] * len(i))

    return input_ids, attention_mask
    

def get_input_feature_text(example, tokenizer_text, max_length):
    choices_inputs = []

    for idx, ending in enumerate(example['endings']):
        context = example['context']
        question = example['question']
        inputs = tokenizer_text(
            context,
            question + ending ,
            add_special_tokens=True,
        )
        choices_inputs.append(inputs)
    
    input_ids = [x["input_ids"] for x in choices_inputs]
    attention_mask = (
        [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
    )
    token_type_ids = (
        [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
    )

    return input_ids, attention_mask, token_type_ids

def amr_batch_encode(tokenizer, input_lst):
    res = []
    for itm_lst in input_lst:
        res.append(
            get_ids(tokenizer, itm_lst)
        )

    return res

def get_ids(tokenizer, tokens):
    token_ids = [tokenizer.encoder.get(b, tokenizer.unk_token_id) for b in tokens]
    return token_ids


@dataclass
class CustomDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    amr_tokenizer: PreTrainedTokenizerBase
    text_tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        
        amr = []
        text = []
        for i in features:
            for id, _ in enumerate(i['amr_input_ids']):
                amr.append({
                    'input_ids': i['amr_input_ids'][id],
                    'attention_mask' : i['amr_attention_mask'][id],
                    'label':i['label']
                })
                text.append({
                    'input_ids': i['text_input_ids'][id],
                    'attention_mask' : i['text_attention_mask'][id],
                    'token_type_ids': i['text_token_type_ids'][id],
                    'label':i['label']
                })

        batch_amr = self.amr_tokenizer.pad(
            amr,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch_size = len(features)
        num_choices = len(features[0]['amr_input_ids'])
        input_len_amr = len(batch_amr['input_ids'][0])

        batch={}
        batch['amr_input_ids'] = torch.reshape(batch_amr['input_ids'] , (batch_size, num_choices, input_len_amr))
        batch['amr_attention_mask'] = torch.reshape(batch_amr['attention_mask'] , (batch_size, num_choices, input_len_amr))

        batch_text = self.text_tokenizer.pad(
            text,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        input_len_text = len(batch_text['input_ids'][0])
        batch['text_input_ids'] = torch.reshape(batch_text['input_ids'] , (batch_size, num_choices, input_len_text))
        batch['text_attention_mask'] = torch.reshape(batch_text['attention_mask'] , (batch_size, num_choices, input_len_text))
        batch['text_token_type_ids'] = torch.reshape(batch_text['token_type_ids'] , (batch_size, num_choices, input_len_text))

        indices = []
        for idx, v in enumerate(batch_amr['label']):
            if (idx % num_choices) != 0:
                indices.append(idx)

        batch['labels'] = th_delete(batch_amr['label'], indices)

        return batch   

def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


            #    max_length=max_length,
            #padding="max_length",
            #truncation=True,