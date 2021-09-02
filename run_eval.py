import argparse
import json
import pickle
from pathlib import Path

import torch
# import wandb
from tqdm import tqdm

from transformers_local import AutoModelForSeq2SeqLM, AutoTokenizer

from my_utils import choose_gpu, flatten_constraints

try:
    from .utils import calculate_rouge, calculate_rouge_new, use_task_specific_params, calculate_bleu_score, trim_batch
except ImportError:
    from utils import calculate_rouge, calculate_rouge_new, use_task_specific_params, calculate_bleu_score, trim_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def generate_summaries_or_translations(
        examples: list,
        out_file: str,
        model_name: str,
        constraint_path: str,
        batch_size: int = 8,
        device: str = 'cuda',
        fp16=False,
        task="summarization",
        decoder_start_token_id=None,
        use_DBA=False,
        use_rl=False,
        max_tgt_length=32,
        num_beams=1,
        **gen_kwargs,
) -> None:
    if use_rl:
        assert num_beams == 1
    elif use_DBA:
        assert num_beams > 1
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()
    if decoder_start_token_id is None:
        decoder_start_token_id = gen_kwargs.pop("decoder_start_token_id", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # update config with summarization specific params
    if use_DBA or use_rl:
        f = open(constraint_path)
        all_constraints = json.load(f)
    # use_task_specific_params(model, task)
    cur_pos = 0
    keyword_logits_l = []
    input_ids_l = []
    for batch in tqdm(list(chunks(examples, batch_size))):
        if "t5" in model_name:
            batch = [model.config.prefix + text for text in batch]
        cur_size = len(batch)
        batch = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(device)
        input_ids, attention_mask = trim_batch(**batch, pad_token_id=tokenizer.pad_token_id)

        if use_DBA or use_rl:
            gen_kwargs['constraints'] = []
            for i in range(cur_pos, cur_pos + cur_size):
                # print(all_constraints[i])
                if all_constraints[i] != []:
                    gen_kwargs['constraints'].append(
                        tokenizer(all_constraints[i], add_special_tokens=False)["input_ids"])
                else:
                    gen_kwargs['constraints'].append([])
            if use_rl:
                gen_kwargs['constraints'] = flatten_constraints(gen_kwargs['constraints'])
                # dummy parameter
                gen_kwargs['gold_constraints'] = gen_kwargs['constraints']

        with torch.no_grad():
            summaries = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=decoder_start_token_id,  # TODO double-check the actual id in use is 0 or 2
                max_length=max_tgt_length,
                use_rl=use_rl,
                do_sample=use_rl,  # do_sample controls whether use threshold now
                num_beams=num_beams,
                **gen_kwargs,
            )
        if num_beams == 1:
            summaries = summaries[0]  # .generate() return a tuple now
        cur_pos += cur_size
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


def run_generate():
    parser = argparse.ArgumentParser()
    dataset = 'QG'
    model_path = f'{dataset}_large'
    split = 'test'
    save_path_suffix = '.run0'
    # constraint_file = 'test.available_gold_constraints.json'
    constraint_file = 'constraint_kpe_em.json'

    # use_copy has to be the same as in training
    parser.add_argument("--use_copy", default=False, help='use copy mechanism')
    # use larger num_beams when use_DBA=True
    parser.add_argument("--num_beams", type=int, default=20, required=False, help="beam size")
    parser.add_argument("--max_tgt_length", type=int, default=20, required=False, help="max target length")
    parser.add_argument("--bs", type=int, default=6, required=False, help="batch size")

    # DBA related parameters
    parser.add_argument("--use_DBA", default=True, help='use DBA. Only work when num_beams > 1')
    parser.add_argument("--use_rl", default=False, help='use rl constrained. Only work when num_beams = 1')
    parser.add_argument("--rl_mode", type=str, default='constraint_logits')
    parser.add_argument("--partial", default=False, help='use DDBA filtering, only partial constraints are met.')
    parser.add_argument("--partial_top_k", default=5, help='keep constraint if it is in top-k tokens')
    parser.add_argument("--partial_top_p", default=None, help='keep constraint if it is in top_p prob mass')
    parser.add_argument("--partial_min_score", default=None, help='keep constraint if its score is higher than this')
    parser.add_argument("--partial_score_mode", default=0,
                        help='which score to use for filtering, 0 use softmax vocab_prob, 1 use softmax copy_prob, '
                             '2/3 use self-attention scores (static)')

    ### below should not be changed usually
    parser.add_argument("--dataset", default=dataset, type=str)
    parser.add_argument("--constraint_file", default=constraint_file, type=str)
    parser.add_argument("--use_wandb", default=False)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=f'test_data/{dataset}/{split}.source')
    parser.add_argument("--reference_path", type=str, required=False, default=f'test_data/{dataset}/{split}.target')
    parser.add_argument("--save_path", type=str, help="where to save pred", default=None)
    parser.add_argument("--save_path_suffix", type=str, help="where to save pred", default=save_path_suffix)
    parser.add_argument("--score_path", type=str, required=False, default=None)
    parser.add_argument("--constraint_path", type=str, required=False, default=None)
    parser.add_argument("--task", type=str, default="summarization", help="typically translation or summarization")
    parser.add_argument(
        "--decoder_start_token_id",
        type=int,
        default=None,
        required=False,
        help="decoder_start_token_id (otherwise will look at config)",
    )
    parser.add_argument("--n_obs", type=int, default=-1, help="How many observations. Defaults to all.")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    if args.model_name is None:
        if args.model_path is not None:
            model_path = args.model_path
        args.model_name = f'output/{model_path}/best_tfmr'
    if args.save_path is None:
        args.save_path = f'output/{model_path}/{split}.pred{args.save_path_suffix}'
    if args.score_path is None:
        args.score_path = f'output/{model_path}/{split}_score{args.save_path_suffix}.json'
    if args.constraint_path is None:
        args.constraint_path = f"test_data/{args.dataset}/{args.constraint_file}"

    if args.use_DBA:
        args.save_path += '.dba'
    if Path(args.save_path).exists() and 'del' not in args.save_path:
        print(f'[ERROR] {args.save_path} exists')
        exit()
    examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]
    if args.n_obs > 0:
        examples = examples[: args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(project="EDE", config=args, sync_tensorboard=True)
        wandb.run.name = '[test]' + args.save_path

    print(f'use_copy={args.use_copy}, use_DBA={args.use_DBA}, num_beams={args.num_beams}')
    print(f'max_tgt_length={args.max_tgt_length}, save_path={args.save_path} \n')
    if args.use_DBA:
        print(f'constraint_path={args.constraint_path}')
        if args.use_rl:
            print('use rl constrained decoding')
        else:
            print(f'partial={args.partial}')
            if args.partial:
                print(f'partial_top_k={args.partial_top_k}, partial_top_p={args.partial_top_p}, '
                      f'partial_min_score={args.partial_min_score}, partial_score_mode={args.partial_score_mode}\n')

    generate_summaries_or_translations(
        examples,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        use_copy=args.use_copy,
        use_DBA=args.use_DBA,
        use_rl=args.use_rl,
        fp16=args.fp16,
        task=args.task,
        decoder_start_token_id=args.decoder_start_token_id,
        constraint_path=args.constraint_path,
        num_beams=args.num_beams,
        max_tgt_length=args.max_tgt_length,
        partial=args.partial,
        partial_top_k=args.partial_top_k,
        partial_top_p=args.partial_top_p,
        partial_min_score=args.partial_min_score,
        partial_score_mode=args.partial_score_mode,
        rl_mode=args.rl_mode,
    )

    if args.reference_path is None:
        return
    # Compute scores
    score_fn = calculate_bleu_score if "translation" in args.task else calculate_rouge_new
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][: len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    print(scores)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w+"))
    return scores


if __name__ == "__main__":
    choose_gpu(min_gpu_memory=4000)
    run_generate()
