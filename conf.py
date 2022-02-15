from transformers_local.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    # get_polynomial_decay_schedule_with_warmup,
)

# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    # "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


def add_generic_args(parser, root_dir):
    #  allow all pl args? parser = pl.Trainer.add_argparse_args(parser)
    dataset = 'QG'
    remark = f'{dataset}'
    parser.add_argument("--remark", type=str, default=remark)
    parser.add_argument("--rl_mode", type=str, default='feat', help='not used in the current project')
    parser.add_argument("--use_copy", default=False, help='use copy mechanism. Only work when num_beams > 1 for now.')
    parser.add_argument("--data_dir", default=f'test_data/{dataset}', type=str)
    parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")

    parser.add_argument("--num_train_epochs", dest="max_epochs", default=200, type=int)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--eval_batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--ckpt_metric", type=str, default='rougeL')
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, required=False)
    parser.add_argument("--val_check_interval", default=1.0, type=float, help='check every k steps or k% in an epoch')

    parser.add_argument(
        "--max_source_length",
        default=256,  # 90~95%
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--val_max_target_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--test_max_target_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    ### NB. below should not be changed usually ###
    parser.add_argument(
        "--output_dir",
        default=f'output/{remark}',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        default=False,
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", default=True, action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--debug", default=False, action="store_true", help="overfit on training set")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--model_name_or_path",
        default='facebook/bart-base',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--encoder_layerdrop",
        type=float,
        help="Encoder layer dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--decoder_layerdrop",
        type=float,
        help="Decoder layer dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--dropout", type=float, help="Dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--attention_dropout", type=float, help="Attention dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--lr_scheduler",
        default="linear",
        choices=arg_to_scheduler_choices,
        metavar=arg_to_scheduler_metavar,
        type=str,
        help="Learning rate scheduler",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")

    parser.add_argument("--freeze_embeds", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--freeze_copy", action="store_true")
    parser.add_argument("--sortish_sampler", action="store_true", default=False)
    parser.add_argument("--logger_name", default='wandb', type=str, choices=["default", "wandb", "wandb_shared"])
    parser.add_argument("--task", type=str, default="summarization", required=False)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument("--src_lang", type=str, default="", required=False)
    parser.add_argument("--tgt_lang", type=str, default="", required=False)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=-1,
        required=False,
        help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
    )

    parser.add_argument("--fast_dev_run", default=False)
    return parser
