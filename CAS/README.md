
The most important files in the codebase are described below. 

`train.py`: the entry point, set the constraint type by `-constraints`.

`prepro/entity_masking.py`: used for entity finding and constraint creation.

`models/DBA.py`: the dynamic beam allocation algorithm.

`models/predictor.py`: the functions that conduct constrained decoding.

For keyphrase extraction, we adapted the code from https://github.com/thunlp/BERT-KPE.

For the base model BERTSum, we modified the code from https://github.com/nlpyang/PreSumm.
