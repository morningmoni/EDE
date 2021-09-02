import argparse
import glob
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader

from conf import add_generic_args
from lightning_base import BaseTransformer, generic_train
from transformers_local import MarianTokenizer, MBartTokenizer, T5ForConditionalGeneration

from my_utils import choose_gpu
from transformers_local.modeling_bart import shift_tokens_right

try:
    from .utils import (
        assert_all_frozen,
        use_task_specific_params,
        lmap,
        flatten_list,
        pickle_save,
        save_git_info,
        save_json,
        freeze_params,
        calculate_rouge,
        get_git_info,
        ROUGE_KEYS,
        calculate_bleu_score,
        Seq2SeqDataset,
        TranslationDataset,
        label_smoothed_nll_loss,
    )

    from .callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
except ImportError:
    from utils import (
        Seq2SeqDataset,
        TranslationDataset,
        assert_all_frozen,
        use_task_specific_params,
        lmap,
        flatten_list,
        pickle_save,
        save_git_info,
        save_json,
        freeze_params,
        calculate_rouge,
        get_git_info,
        ROUGE_KEYS,
        calculate_bleu_score,
        label_smoothed_nll_loss,
    )
    from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback

logger = logging.getLogger(__name__)


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    use_kw = False  # not used in the project
    if use_kw:
        loss_names = ["loss", "s2s_loss", "keyword_loss", "keyword_acc", "keyword_f1"]
    else:
        loss_names = ["loss"]
    metric_names = ROUGE_KEYS

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        # use_task_specific_params(self.model, "summarization")
        # save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.val_metric = hparams.ckpt_metric
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        if isinstance(self.tokenizer, MBartTokenizer) or isinstance(self.tokenizer, MarianTokenizer):
            self.dataset_class = TranslationDataset
        else:
            self.dataset_class = Seq2SeqDataset

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        if self.use_kw:
            keyword_labels = batch["keyword_labels"]
        else:
            keyword_labels = None

        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(target_ids)
            lm_labels = target_ids
        else:
            # shift lm vs shift input
            # decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
            # lm_labels = target_ids[:, 1:].clone()  # why clone?
            decoder_input_ids = shift_tokens_right(target_ids, pad_token_id)
            lm_labels = target_ids

        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                       return_dict=self.hparams.use_copy, use_copy=self.hparams.use_copy, keyword_labels=keyword_labels)
        if keyword_labels is not None:
            keyword_pred = torch.sigmoid(outputs['keyword_logits']) > 0.5
            keyword_labels = batch['keyword_labels']
            keyword_acc = torch.mean((keyword_pred == keyword_labels).float())
            keyword_f1 = f1_score(batch['keyword_labels'].cpu().numpy().reshape(-1),
                                  keyword_pred.cpu().numpy().reshape(-1))
            keyword_f1 = torch.FloatTensor([keyword_f1]).type_as(keyword_acc)

        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py
            if self.hparams.use_copy:
                # output is already normalized for copy_mechanism (should get same results as CrossEntropyLoss tho)
                loss_fct = torch.nn.NLLLoss(ignore_index=pad_token_id)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
            lm_logits = outputs[0]
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, lm_labels, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        if 'keyword' in self.val_metric:
            final_loss = outputs['keyword_loss']
        # TODO why 'keyword_loss' not in outputs?
        elif self.use_kw:
            final_loss = loss + outputs['keyword_loss']
        else:
            return (loss,)

        return (final_loss, loss, outputs['keyword_loss'], keyword_acc, keyword_f1)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["decoder_input_ids"].ne(self.pad).sum()
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        rouges = {k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]}
        rouges.update({k: v.item() for k, v in losses.items()})
        rouge_tensor: torch.FloatTensor = torch.tensor(rouges[self.val_metric]).type_as(loss)
        losses.update(rouges)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {"log": metrics, "preds": preds, f"{prefix}_loss": loss, f"{prefix}_{self.val_metric}": rouge_tensor}

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        batch_size = batch["input_ids"].size()[0]
        constraints = [[] for _ in range(batch_size)]
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            num_beams=2,
            decoder_start_token_id=self.decoder_start_token_id,
            use_copy=self.hparams.use_copy,
            constraints=constraints
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        val_path = Path(self.output_dir) / f"val-{self.step_count}.pred"
        if not val_path.exists():
            with open(val_path, 'w') as o:
                for p, t in zip(preds, target):
                    o.write(f'{p} ***** {t}\n')

        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)
        sampler = None
        if self.hparams.sortish_sampler and type_path == "train":
            assert self.hparams.gpus <= 1  # TODO: assert earlier
            sampler = dataset.make_sortish_sampler(batch_size)
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size,
                                         shuffle=True if not self.hparams.debug else False)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)


class TranslationModule(SummarizationModule):
    mode = "translation"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.dataset_kwargs["src_lang"] = hparams.src_lang
        self.dataset_kwargs["tgt_lang"] = hparams.tgt_lang

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu_score(preds, target)


def main(args, model=None) -> SummarizationModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    # if len(os.listdir(args.output_dir)) > 3 and args.do_train and 'del' not in args.output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if 'del' in args.output_dir:
        args.logger_name = 'default'
    if model is None:
        if args.task == "summarization":
            model: SummarizationModule = SummarizationModule(args)
        else:
            model: SummarizationModule = TranslationModule(args)

    dataset = Path(args.data_dir).name
    if (
            args.logger_name == "default"
            or args.fast_dev_run
            or str(args.output_dir).startswith("/tmp")
            or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project='KC-Gen')

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    cp_file = None
    if args.do_train:
        if (Path(args.output_dir) / 'checkpointlast.ckpt').exists():
            cp_file = str(Path(args.output_dir) / 'checkpointlast.ckpt')
        else:
            cp_files = glob.glob(os.path.join(args.output_dir, '*.ckpt'))
            if len(cp_files) > 0:
                cp_file = sorted(cp_files, key=lambda x: int(x.split('=')[-1][:-5]), reverse=True)[0]
        if cp_file is not None:
            print(f'resume training from {cp_file} ...')

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric) if not args.debug else None,
        early_stopping_callback=es_callback,
        logger=logger,
        resume_cp_file=cp_file,
        # TODO: early stopping callback seems messed up
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args.do_predict:
        return model

    trainer.test()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    parser = add_generic_args(parser, os.getcwd())

    args = parser.parse_args()
    print(args.output_dir)
    choose_gpu()
    main(args)
