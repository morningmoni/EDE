import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast


def create_mlm_labels(wp_ids, wp_tokens, spacy_tokens):
    def get_span(wp_tokens, wp_idx):
        # find the whole token from word pieces
        wp_idx_end = wp_idx + 1
        while wp_idx_end < len(wp_tokens) and wp_tokens[wp_idx_end].startswith('##'):
            wp_idx_end += 1
        return " ".join(wp_tokens[wp_idx:wp_idx_end]).replace(" ##", "").strip(), wp_idx_end

    def match_span(spacy_tokens, token, spacy_idx):
        # match the whole token using spacy's indices
        spacy_idx_end = spacy_idx + 1
        while str(spacy_tokens[spacy_idx:spacy_idx_end]) != token:
            spacy_idx_end += 1
        return spacy_idx_end

    labels = np.array([-100] * len(wp_tokens))
    wp_idx = spacy_idx = 0
    while wp_idx < len(wp_tokens):
        token, wp_idx_end = get_span(wp_tokens, wp_idx)
        if token in ['[CLS]', '[SEP]']:
            wp_idx += 1
            spacy_idx += 3
            continue

        spacy_idx_end = match_span(spacy_tokens, token, spacy_idx)

        all_noent = all(spacy_token.ent_type_ == '' for spacy_token in spacy_tokens[spacy_idx:spacy_idx_end])
        # all_ent = all(spacy_token.ent_type_ != '' for spacy_token in spacy_tokens[spacy_idx:spacy_idx_end])
        # if not (all_noent or all_ent):
        #     print(f'[inconsistent typing]: {spacy_tokens[spacy_idx:spacy_idx_end]} {[spacy_token.ent_type_ for spacy_token in spacy_tokens[spacy_idx:spacy_idx_end]]}')

        if not all_noent:
            labels[wp_idx:wp_idx_end] = wp_ids[wp_idx:wp_idx_end]
        wp_idx = wp_idx_end
        spacy_idx = spacy_idx_end
    return list(labels)


def mask_ent(fname):
    print(fname)
    data = torch.load(fname)
    for d in tqdm(data):
        tokens = tokenizer.convert_ids_to_tokens(d['src'])
        src = " ".join(tokens).replace(" ##", "").strip()
        with nlp.disable_pipes("tagger", "parser"):
            src = nlp(src)
        labels = create_mlm_labels(d['src'], tokens, src)
        d['mlm_labels'] = labels
    torch.save(data, fname)


def mask_ent_gold(fname):
    print(fname)
    data = torch.load(fname)
    for d in tqdm(data):
        tokens = tokenizer.convert_ids_to_tokens(d['tgt'])
        tgt = " ".join(tokens).replace(" ##", "").strip()
        with nlp.disable_pipes("tagger", "parser"):
            tgt = nlp(tgt)
        labels_tgt = create_mlm_labels(d['tgt'], tokens, tgt)
        d['mlm_labels_tgt'] = labels_tgt
        # token ids of entities in ref summ
        ent_ids_in_tgt = set([i for i in labels_tgt if i != -100])
        # mask entity spans that appear in ref summ
        labels_gold = [i if i in ent_ids_in_tgt else -100 for i in d['mlm_labels']]
        d['mlm_labels_gold'] = labels_gold
    torch.save(data, fname)


def count_ent_tgt(fname):
    print(fname)
    data = torch.load(fname)
    mean_l = []
    ct_l = []
    for d in tqdm(data):
        tokens_l = [i for i in d['mlm_labels_tgt'] if i != -100]
        n = list(Counter(tokens_l).values())
        if len(n) == 0:
            mean = 0
        else:
            mean = np.mean(n)
        mean_l.append(mean)
        ct_l.extend(n)
    print(pd.Series(mean_l).describe())
    print(pd.Series(ct_l).describe())


def count_constraints_run(fname):
    # print(fname)
    data = torch.load(fname)
    ct_l = []
    for d in data:
        n_constraints = sum([len(p) for p in d[CONSTRAINT_MODE]])
        ct_l.append(n_constraints)
    # print(pd.Series(ct_l).describe())
    return ct_l


def count_constraints_combine(res):
    print(CONSTRAINT_MODE)
    ct_total = []
    for ct_l in res:
        ct_total.extend(ct_l)
    print('#constraints stats')
    print(pd.Series(ct_total).describe(), '\n')
    return ct_total


def count_constraints(n=10):
    return multi_runs(count_constraints_run, para, count_constraints_combine, n=n)


def chunks(l, n):
    n = len(l) // n
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def multi_runs(f, para, f_combine=None, n=30):
    print(f'len(para) = {len(para)}')
    print(f'n_cpus = {n}')
    pool = Pool(n)
    res = pool.map(f, para)
    if f_combine is not None:
        res = f_combine(res)
    return res


def entity_masking():
    multi_runs(mask_ent, para, n=n_cpus)


def entity_gold_masking():
    multi_runs(mask_ent_gold, para, n=n_cpus)


def extract_phrases(text, ent_only=True, deduplicate=False, return_str=True):
    if type(text) == str:
        # docs.noun_chunks needs "parser"
        if ent_only:
            with nlp.disable_pipes("tagger", "parser"):
                doc = nlp(text)
        else:
            with nlp.disable_pipes("tagger"):
                doc = nlp(text)
    else:
        doc = text
    spans = list(doc.ents)
    if not ent_only:
        spans += list(doc.noun_chunks)

    if deduplicate:
        # only the longest is reserved?
        spans = spacy.util.filter_spans(spans)
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
    #         if return_str:
    #             return [str(span) for span in doc]
    #         return spans

    if return_str:
        # NB lower() in extract_phrases
        return [str(span).lower() for span in spans]
    return spans


def add_constraints_run(fname):
    mode = CONSTRAINT_MODE
    print(fname, mode)
    data = torch.load(fname)
    if mode == 'constraints_ent':
        for d in tqdm(data):
            ref_p = extract_phrases(d['tgt_txt' + RAW], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            # NB. can also allow duplicates
            tokens = list(set(ref_p))[:MAX_NUM_ENT]
            d[mode] = [tokenizer.encode(t) for t in tokens]
    elif mode == 'constraints_ent_miss':
        for d in tqdm(data):
            ref_p = extract_phrases(d['tgt_txt' + RAW], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            sys_p = extract_phrases(d['sys_txt'], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            missing_tokens = list(set(ref_p) - set(sys_p))[:MAX_NUM_ENT]
            d[mode] = [tokenizer.encode(t) for t in missing_tokens]
    elif mode == 'constraints_ent_miss_str':
        for d in tqdm(data):
            ref_p = extract_phrases(d['tgt_txt' + RAW], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            missing_tokens = list({i for i in ref_p if i not in d['sys_txt']})[:MAX_NUM_ENT]
            d[mode] = [tokenizer.encode(t) for t in missing_tokens]
    elif mode == 'constraints_rand4':
        for d in tqdm(data):
            ref_p = nlp.make_doc(d['tgt_txt'])
            tokens = [str(i) for i in ref_p]
            rand4 = random.choices(tokens, k=min(len(tokens), 4))
            d[mode] = [tokenizer.encode(t) for t in rand4]
    elif mode == 'constraints_rand4_miss':
        for d in tqdm(data):
            ref_p = nlp.make_doc(d['tgt_txt'])
            sys_p = nlp.make_doc(d['sys_txt'])
            missing_tokens = set([str(i) for i in ref_p]) - set([str(i) for i in sys_p])
            d[mode] = [tokenizer.encode(t) for t in list(missing_tokens)[:4]]
    elif mode == 'constraints_phr4':
        for d in tqdm(data):
            ref_p = nlp.make_doc(d['tgt_txt'])
            tokens = [str(i) for i in ref_p]
            start_idx = random.randint(0, max(0, len(tokens) - 4))
            phr4 = ' '.join(tokens[start_idx: start_idx + 4])
            d[mode] = [tokenizer.encode(phr4)]
    elif mode == 'constraints_ent_inSrc_str':
        for d in tqdm(data):
            ref_p = extract_phrases(d['tgt_txt' + RAW], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            src = ' '.join(d['src_txt'])
            tokens = {i for i in set(ref_p) if i in src}
            d[mode] = [tokenizer.encode(t) for t in tokens]
    elif mode == 'constraints_ent_miss_inSrc':
        for d in tqdm(data):
            ref_p = extract_phrases(d['tgt_txt' + RAW], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            sys_p = extract_phrases(d['sys_txt'], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            if RAW == '':
                src_p = extract_phrases(' '.join(d['src_txt']), ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            else:
                src_p = extract_phrases(d['src_txt' + RAW], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            missing_tokens = set(ref_p) & set(src_p) - set(sys_p)
            d[mode] = [tokenizer.encode(t) for t in missing_tokens]
    elif mode == 'constraints_ent_miss_inSrc_str':
        for d in tqdm(data):
            ref_p = extract_phrases(d['tgt_txt' + RAW], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            sys_p = extract_phrases(d['sys_txt'], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            src = ' '.join(d['src_txt'])
            missing_tokens = {i for i in set(ref_p) if i in src} - set(sys_p)

            # ref_p2 = extract_phrases(d['tgt_txt'], ent_only=ENT_ONLY, deduplicate=False, return_str=True)
            # missing_tokens2 = {i for i in set(ref_p2) if i in src} - set(sys_p)
            # if len(missing_tokens2) > len(missing_tokens):
            #     _ = 1

            d[mode] = [tokenizer.encode(t) for t in missing_tokens]
    elif mode.startswith('constraints_kpe'):
        key = mode.split('_')[-1]
        for d in tqdm(data):
            if key in d:
                tokens = [' '.join(p) for p in d[key]]
                d[mode] = [tokenizer.encode(t) for t in tokens]
            else:
                d[mode] = []
    else:
        raise NotImplementedError
    torch.save(data, fname)


def add_constraints():
    multi_runs(add_constraints_run, para, n=n_cpus)


def rename(fname):
    print(fname)
    data = torch.load(fname)
    for d in tqdm(data):
        d['constraints_rand4_miss'] = d.pop('constraints')
        d['constraints_ent_miss_inSrc'] = d.pop('constraints_refANDsrc')
        d['constraints_ent_miss_inSrc_NP'] = d.pop('constraints_refANDsrc_NP')
    torch.save(data, fname)


def get_keyphrases_run(data):
    res = []
    for i in tqdm(data):
        kp = extract_phrases(i, ent_only=ENT_ONLY, deduplicate=False, return_str=True)
        res.append(';'.join(kp))
    return res


def get_keyphrases_combine(res):
    res_total = []
    for i in res:
        res_total.extend(i)
    return res_total


def get_keyphrases(str_l):
    para = chunks(str_l, n_cpus)
    return multi_runs(get_keyphrases_run, para, get_keyphrases_combine, n=n_cpus)


def keyphrase_data_write(dataset='xsum'):
    path = Path(f'KGSum/BART/data/{dataset}/')
    out_path = Path(f'BERT-KPE/data/dataset/{dataset}')
    for mode in ['val', 'test']:
        with open(path / f'{mode}.source') as f:
            src_l = f.readlines()
        with open(path / f'{mode}.target') as f:
            tgt_l = f.readlines()
        keyphrases_tgt_l = get_keyphrases(tgt_l)
        with open(out_path / f'{dataset}_{mode}.json', 'w') as o:
            for src, kp in zip(src_l, keyphrases_tgt_l):
                data = {}
                data['abstract'] = src
                data['title'] = ''
                data['keyword'] = kp
                o.write(json.dumps(data) + '\n')


DATASET = 'xsum'
DATA_PATH = Path('KGSum/PreSumm/data')
para = [fname for fname in (DATA_PATH / DATASET).glob(f'{DATASET}.test2.3.bert.pt')]
# para = [fname for fname in (DATA_PATH / DATASET).glob(f'{DATASET}.test.*.bert.pt')]
# para.extend([fname for fname in (DATA_PATH / DATASET).glob(f'{DATASET}.train.*.bert.pt')])
# para.extend([fname for fname in (DATA_PATH / DATASET).glob(f'{DATASET}.valid.*.bert.pt')])
n_cpus = 10

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', add_special_tokens=False)
model = "en_core_web_lg"
# model = "en"
print('spacy loading', model)
nlp = spacy.load(model)

MAX_NUM_ENT = 100

# use lowercase or original case
RAW = '_raw'
assert RAW in ['', '_raw']

ENT_ONLY = False


# CONSTRAINT_MODE = 'constraints_phr4'
# data = torch.load(para[0])
# for d in tqdm(data[:100]):
#     print(tokenizer.decode(d[CONSTRAINT_MODE][0]))
#     # raise

CONSTRAINT_MODE_l = ['constraints_ent_miss']
# CONSTRAINT_MODE_l = ['constraints_phr4', 'constraints_rand4', 'constraints_rand4_miss', 'constraints_ent',
#                      'constraints_ent_miss', 'constraints_ent_miss_inSrc', 'constraints_ent_miss_inSrc_NP']
# CONSTRAINT_MODE_l = ['constraints_kpe2.4', 'constraints_kpe', 'constraints_kpe3.5', 'constraints_kpe4',
#                      'constraints_kpe1.5']
# CONSTRAINT_MODE_l = ['constraints_kpe-rank3', 'constraints_kpe-rank6', 'constraints_kpe-rank9']
for CONSTRAINT_MODE in CONSTRAINT_MODE_l:
    # for p in para:
    #     add_constraints_run(p)
    add_constraints()
    count_constraints()

if RAW == '':
    print('using lowercase text')
else:
    print('using original cased text')

print(f'ENT_ONLY={ENT_ONLY}')
