import argparse
import os
import time

from utils.utils import get_gpu_memory_map

parser = argparse.ArgumentParser()
parser.add_argument('-min_gpu_memory', default=11000, type=int)
parser.add_argument('-max_n_gpus', default=2, type=int)
parser.add_argument('-remark', default='ent_2gpus', help='used in the log file name')
args = parser.parse_args()

while True:
    gpus = []
    sorted_gpu_info = get_gpu_memory_map()
    for gpu_id, (mem_left, util) in sorted_gpu_info:
        if mem_left >= args.min_gpu_memory:
            gpus.append(gpu_id)
            print(f'use gpu:{gpu_id} with {mem_left} MB left, util {util}%')
        if len(gpus) == args.max_n_gpus:
            print('[max num of gpus reached]')
            break
    if len(gpus) != args.max_n_gpus:
        print(f'cannot find {args.max_n_gpus} gpus with memory left >= {args.min_gpu_memory} MB, sleep 600s...')
        time.sleep(600)
    else:
        os.system(
            f'python train.py -min_gpu_memory {args.min_gpu_memory} -max_n_gpus {args.max_n_gpus} -remark {args.remark}')
        break
