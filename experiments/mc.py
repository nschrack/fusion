#!/usr/bin/env python
# coding=utf-8

""" Finetuning models on CaseHOLD (e.g. Bert, RoBERTa, LEGAL-BERT)."""

import sys
import os
import torch

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
)
from transformers.trainer_utils import is_main_process
from dataloader_fusion import MultipleChoiceDataset
from dataloader_fusion import Split
from sklearn.metrics import f1_score
from models.bart import BartForMultipleChoiceClassificationSentRep as BartForMultipleChoice
from models.fusion import Fusion
from spring.spring_amr.tokenization_bart import PENMANBartTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.utils.data import DataLoader

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


class ClassificationTask(pl.LightningModule):
    def __init__(self, model, lr, batch_size, train_dataset):
        super().__init__()
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.train_dataset = train_dataset

    def training_step(self, batch, batch_idx):
        ti, ta, tt, ai, at, y = batch
        loss, logits = self.model(ti, ta, tt, ai, at, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx)
        metrics["val_loss"] = loss
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx)
        metrics["test_loss"] = loss
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        ti, ta, tt, ai, at, y = batch
        loss, logits = self.model(ti, ta, tt, ai, at, y)
        metrics = self.compute_metrics(logits, y)

        return loss, metrics

    def compute_metrics(self, logits, labels):
        preds = np.argmax(logits, axis=1)
        # Compute macro and micro F1 for 5-class CaseHOLD task
        macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro', zero_division=0)
        return {'macro-f1': macro_f1, 'micro-f1': micro_f1}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ti, ta, tt, ai, at, y = batch
        y_hat = self.model(ti, ta, tt, ai, at, y)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)



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
    pl.seed_everything(training_args.seed)

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
    """
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
    """
    task = ClassificationTask(
        model,
        training_args.learning_rate, 
        training_args.per_device_train_batch_size,
        train_dataset
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        auto_scale_batch_size=True,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        max_epochs=training_args.num_train_epochs,
        deterministic=True,
        #accelerator="gpu", devices=0, precision=16

        ) 
    
    #Trainer(accelerator="gpu", devices=2)
    #trainer.fit(task, val_dataloaders=DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size))
    trainer.test(dataloaders=predict_dataset)

    """
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
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
    """

if __name__ == "__main__":
    main()
