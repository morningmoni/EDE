import os
import subprocess
import time
from datetime import datetime


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_info = [eval(x) for x in result.strip().split('\n')]
    gpu_info = dict(zip(range(len(gpu_info)), gpu_info))
    sorted_gpu_info = sorted(gpu_info.items(), key=lambda kv: kv[1][0], reverse=True)
    sorted_gpu_info = sorted(sorted_gpu_info, key=lambda kv: kv[1][1])
    return sorted_gpu_info


def choose_gpu(n_gpus=1, min_gpu_memory=6000, retry=False, sleep_time=30):
    start_time = time.time()
    sorted_gpu_info = get_gpu_memory_map()
    gpustat = subprocess.check_output(
        [
            'gpustat'
        ], encoding='utf-8')
    print(gpustat)
    print(f'gpu_id, (mem_left, util): {sorted_gpu_info}')
    while True:
        gpus = []
        for gpu_id, (mem_left, util) in sorted_gpu_info:
            if mem_left >= min_gpu_memory:
                gpus.append(gpu_id)
                print('use gpu:{} with {} MB left, util {}%'.format(gpu_id, mem_left, util))
            if len(gpus) == n_gpus:
                # print('max num of gpus reached.')
                break
        if len(gpus) == 0:
            if retry:
                print(f'[{datetime.now().strftime("%H:%M:%S")}'
                      f' waited {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}]'
                      f' no gpu has memory >= {min_gpu_memory} MB, sleep {sleep_time}s...', end='\r')
                time.sleep(sleep_time)
            else:
                print(f'no gpu has memory >= {min_gpu_memory} MB, exiting...')
                exit()
        else:
            break
        sorted_gpu_info = get_gpu_memory_map()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    visible_gpus = ','.join([str(gpu_id) for gpu_id in gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus


def flatten_constraints(constraints):
    return [[token for span in sample for token in span] for sample in constraints]
