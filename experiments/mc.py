#!/usr/bin/env python
# coding=utf-8

""" Finetuning models on CaseHOLD (e.g. Bert, RoBERTa, LEGAL-BERT)."""

from lib2to3.pgen2 import token
import sys
import os
import torch
from pynvml import *

# adding case hold home directories to path for imports 
sys.path.insert(0, os.getenv('HOME_PATH'))

# deactivate wandb
os.environ["WANDB_DISABLED"] = "true"

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import random

import transformers
from transformers import (
	AutoConfig,
	AutoModelForMultipleChoice,
	AutoTokenizer,
	EvalPrediction,
	HfArgumentParser,
	Trainer,
	TrainingArguments,
	set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers import EarlyStoppingCallback
from dataloader_fusion import MultipleChoiceDataset, CustomDataCollatorWithPadding
from dataloader_fusion import Split
from sklearn.metrics import f1_score

from models.bart import BartForMultipleChoiceClassificationSentRep as BartForMultipleChoice
from models.fusion import Fusion
from spring.spring_amr.tokenization_bart import PENMANBartTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path_text: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	model_name_or_path_amr: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""
	data_set_path_text: str = field(default="", metadata={"help": "The path to the text data set (if locally)"})
	data_set_path_amr: str = field(default="", metadata={"help": "The path to the amr data set (if locally)"})

	task_name: str = field(default="case_hold", metadata={"help": "The name of the task to train on"})
	max_seq_length_text: int = field(
		default=256,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	max_seq_length_amr: int = field(
		default=1024,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	pad_to_max_length: bool = field(
		default=True,
		metadata={
			"help": "Whether to pad all samples to `max_seq_length`. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch."
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			"value if set."
		},
	)
	max_predict_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
			"value if set."
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
	)
	question: bool = field(
		default=False, metadata={"help": "The data set has a context and question."}
	)
	
def print_gpu_utilization():
	if torch.cuda.is_available():
		nvmlInit()
		handle = nvmlDeviceGetHandleByIndex(0)
		info = nvmlDeviceGetMemoryInfo(handle)
		logger.info(f"GPU memory total: {info.total//1024**2} MB.")
		logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")
	else: 
		logger.info("No cuda device found")

def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	# Add custom arguments for computing pre-train loss
	parser.add_argument("--ptl", type=bool, default=False)
	model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

	if (
		os.path.exists(training_args.output_dir)
		and os.listdir(training_args.output_dir)
		and training_args.do_train
		and not training_args.overwrite_output_dir
	):
		raise ValueError(
			f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
		)

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
	)

	print_gpu_utilization()

	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		training_args.local_rank,
		training_args.device,
		training_args.n_gpu,
		bool(training_args.local_rank != -1),
		training_args.fp16,
	)
	# Set the verbosity to info of the Transformers logger (on main process only):
	if is_main_process(training_args.local_rank):
		transformers.utils.logging.set_verbosity_info()
		transformers.utils.logging.enable_default_handler()
		transformers.utils.logging.enable_explicit_format()
	logger.info("Training/evaluation parameters %s", training_args)

	# Set seed
	set_seed(training_args.seed)

	# Load pretrained model and tokenizer

	tokenizer_amr = PENMANBartTokenizer.from_pretrained(
			'facebook/bart-base', 
			#model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			collapse_name_ops=False,
			use_pointer_tokens=True,
			raw_graph=False,
		)
	tokenizer_text = AutoTokenizer.from_pretrained(
			model_args.model_name_or_path_text,
			cache_dir=model_args.cache_dir,
			# Default fast tokenizer is buggy on CaseHOLD task, switch to legacy tokenizer
			use_fast=True,
		)

	amr_config = AutoConfig.from_pretrained(
		model_args.model_name_or_path_amr,
		num_labels=5,
		finetuning_task=data_args.task_name,
		cache_dir=model_args.cache_dir,
	)

	text_model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_args.model_name_or_path_text) 
	amr_model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_args.model_name_or_path_amr) 

	model = Fusion(
		text_model = text_model,
		amr_model = amr_model,
		concat_emb_dim = 1536,	 # last hidden state size of both models added up
		classifier_dropout = 0.1,
		amr_eos_token_id = amr_config.eos_token_id
	)

	train_dataset = None
	eval_dataset = None

	# If do_train passed, train_dataset by default loads train split from file named train.csv in data directory
	if training_args.do_train:
		train_dataset = \
			MultipleChoiceDataset(
				data_args=data_args,
				tokenizer_text=tokenizer_text,
				tokenizer_amr=tokenizer_amr,
				task=data_args.task_name,
				overwrite_cache=data_args.overwrite_cache,
				mode=Split.train,
			)

	# If do_eval or do_predict passed, eval_dataset by default loads dev split from file named dev.csv in data directory
	if training_args.do_eval:
		eval_dataset = \
			MultipleChoiceDataset(
				data_args=data_args,
				tokenizer_text=tokenizer_text,
				tokenizer_amr=tokenizer_amr,
				task=data_args.task_name,
				overwrite_cache=data_args.overwrite_cache,
				mode=Split.dev,
			)

	if training_args.do_predict:
		predict_dataset = \
			MultipleChoiceDataset(
				data_args=data_args,
				tokenizer_text=tokenizer_text,
				tokenizer_amr=tokenizer_amr,
				task=data_args.task_name,
				overwrite_cache=data_args.overwrite_cache,
				mode=Split.test,
			)

	if training_args.do_train:
		if data_args.max_train_samples is not None:
			train_dataset = train_dataset[:data_args.max_train_samples]
		# Log a few random samples from the training set:
		for index in random.sample(range(len(train_dataset)), 3):
			logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

	if training_args.do_eval:
		if data_args.max_eval_samples is not None:
			eval_dataset = eval_dataset[:data_args.max_eval_samples]

	if training_args.do_predict:
		if data_args.max_predict_samples is not None:
			predict_dataset = predict_dataset[:data_args.max_predict_samples]

	# Define custom compute_metrics function, returns macro F1 metric for CaseHOLD task
	def compute_metrics(p: EvalPrediction):
		preds = np.argmax(p.predictions, axis=1)
		# Compute macro and micro F1 for 5-class CaseHOLD task
		macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
		micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
		return {'macro-f1': macro_f1, 'micro-f1': micro_f1}

	collate_fn = CustomDataCollatorWithPadding(
		tokenizer_amr,
		tokenizer_text,
		pad_to_multiple_of=8 if training_args.fp16 else None,
	)

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=compute_metrics,
		data_collator=collate_fn,
		callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
	)

	# Training
	if training_args.do_train:
		trainer.train(
			model_path=model_args.model_name_or_path_text if os.path.isdir(model_args.model_name_or_path_text) else None
		)
		trainer.save_model()

	# Evaluation on eval_dataset
	if training_args.do_eval:
		logger.info("*** Evaluate ***")
		metrics = trainer.evaluate(eval_dataset=eval_dataset)

		max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
		metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", metrics)

	# Predict on eval_dataset
	if training_args.do_predict:
		logger.info("*** Predict ***")

		predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

		max_predict_samples = (
			data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
		)
		metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

		trainer.log_metrics("predict", metrics)
		trainer.save_metrics("predict", metrics)

		output_predict_file = os.path.join(training_args.output_dir, "test_predictions.csv")
		if trainer.is_world_process_zero():
			with open(output_predict_file, "w") as writer:
				for index, pred_list in enumerate(predictions):
					pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
					writer.write(f"{index}\t{pred_line}\n")

	# Clean up checkpoints
	#checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
	#for checkpoint in checkpoints:
	#	shutil.rmtree(checkpoint)


if __name__ == "__main__":
	main()
