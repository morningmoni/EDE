#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from pathlib import Path

import wandb

from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext
from utils.logging import init_logger
from utils.utils import get_gpu_memory_map

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-remark', default='kpe-rank3-cut1.6', help='used in the log file name')
    parser.add_argument('-constraints', default='constraints_kpe-rank3')
    # parser.add_argument('-constraints', default=None)
    parser.add_argument("-dataset", default='cnndm', type=str, choices=['xsum', 'cnndm'])
    parser.add_argument("-mode", default='test', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-decode_mode", default='normal', type=str,
                        choices=['normal', 'ent', 'ent_gold', 'ent_gold_once'])
    parser.add_argument("-train_from", default='../models/bertsumextabs_xsum_final_model/model_step_30000.pt')
    # parser.add_argument("-train_from", default='../models/bertsumextabs_cnndm_final_model/model_step_148000.pt')
    # parser.add_argument("-train_from", default='../models/cnndm_ent_2gpus/model_step_154000.pt')

    # parser.add_argument("-test_from", default='../models/ent_mlm/model_step_165000.pt')
    # parser.add_argument("-test_from", default='../models/bertsumextabs_xsum_final_model/model_step_30000.pt')
    parser.add_argument("-test_from", default='../models/bertsumextabs_cnndm_final_model/model_step_148000.pt')

    # xsum 32 -> 8.4G, 40 -> 8.3G+, 50 -> 11.0G
    # cnndm 50 -> 8.4G, 60 -> 9.8G
    parser.add_argument("-batch_size", default=50, type=int,
                        help='not num_samples in a batch! but num_samples x tgt_length')
    # test 500 -> 2.4G, 1500 -> 6G and slower
    parser.add_argument("-test_batch_size", default=500, type=int)
    parser.add_argument("-alpha", default=0.9, type=float, help='length normalization')
    parser.add_argument("-beam_size", default=10, type=int)

    parser.add_argument("-save_checkpoint_steps", default=2000, type=int)
    parser.add_argument("-accum_count", default=10, type=int)
    parser.add_argument("-report_every", default=100, type=int)
    parser.add_argument("-train_steps", default=200000, type=int)

    parser.add_argument('-min_gpu_memory', default=3000, type=int)
    parser.add_argument('-max_n_gpus', default=1, type=int)
    parser.add_argument('-seed', default=666, type=int)

    # parameters below are unchanged
    parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-1, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int,
                        help='looks like this many samples are .backward() together in a loop')
    parser.add_argument("-min_length", default=20, type=int)
    parser.add_argument("-max_length", default=100, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=20000, type=int)
    parser.add_argument("-warmup_steps_dec", default=10000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    # NB. file path
    args.log_file = f'../logs/{args.mode}_{args.dataset}_{args.remark}.log'
    args.bert_data_path = Path(f'../data/{args.dataset}')
    args.model_path = f'../models/{args.dataset}_{args.remark}/'
    args.result_path = f'../res/{args.dataset}_{args.remark}'

    if args.dataset == 'cnndm':
        args.min_length = 50
        args.max_length = 200
        args.alpha = .6

    wandb.init(project="KGSum-PreSumm", config=args, sync_tensorboard=True)
    wandb.run.save()
    wandb.run.name = args.remark + ' | ' + wandb.run.name
    wandb.run.save()

    gpus = []
    sorted_gpu_info = get_gpu_memory_map()
    for gpu_id, (mem_left, util) in sorted_gpu_info:
        if mem_left >= args.min_gpu_memory:
            gpus.append(gpu_id)
            print('use gpu:{} with {} MB left, util {}%'.format(gpu_id, mem_left, util))
        if len(gpus) == args.max_n_gpus:
            print('max num of gpus reached.')
            break
    if len(gpus) == 0:
        print(f'no gpu has memory left >= {args.min_gpu_memory} MB, exiting...')
        exit()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    args.visible_gpus = ','.join([str(gpu_id) for gpu_id in gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    args.gpu_ranks = list(range(len(gpus)))
    args.world_size = len(args.gpu_ranks)

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    print('*' * 5, args.log_file)
    if args.mode == 'train':
        print('batch size:', args.batch_size)
        print('accum count:', args.accum_count)
    else:
        print('length penalty:', args.alpha)
        print('beam size:', args.beam_size)
    if args.task == 'abs':
        if args.mode == 'train':
            train_abs(args, device_id)
        elif args.mode == 'validate':
            validate_abs(args, device_id)
        elif args.mode == 'lead':
            baseline(args, cal_lead=True)
        elif args.mode == 'oracle':
            baseline(args, cal_oracle=True)
        if args.mode == 'test':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step)
        elif args.mode == 'test_text':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)

    elif args.task == 'ext':
        if args.mode == 'train':
            train_ext(args, device_id)
        elif args.mode == 'validate':
            validate_ext(args, device_id)
        if args.mode == 'test':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
        elif args.mode == 'test_text':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)
    print('*' * 5, args.log_file)
