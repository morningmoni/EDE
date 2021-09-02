## Code

The code is based on the [seq2seq examples](https://github.com/huggingface/transformers/tree/master/examples) of huggingface transformers (3.0 <= version < 4.0)



The most important files are as follows:

`DDBA.py`: core functions for constrained generation, including a PyTorch implementation of DBA [1], adapted from the official [MXNet implementation](https://github.com/awslabs/sockeye/blob/master/sockeye/lexical_constraints.py)

`finetune.py`: model training

`run_eval.py`: model inference with or without constraints

`transformers_local/generation_utils.py`: modified the functions related to model decoding for enforcing constraints

`transformers_local/modeling_bart.py`: implemented BART+copy mechanism and other side functions



[1] "Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation", NAACL 2018

