#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function

import codecs
import math

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.DBA import dba_topk, ConstrainedHypothesis, init_constraints
from translate.beam import GNMTGlobalScorer
from utils.utils import rouge_results_to_str, test_rouge, tile


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


# TODO for debugging only
from pytorch_transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src = translation_batch["predictions"], translation_batch["scores"], \
                                                      translation_batch["gold_score"], batch.tgt_str, batch.src

        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##', '')
            gold_sent = ' '.join(tgt_str[b].split())
            # translation = Translation(fname[b],src[:, b] if src is not None else None,
            #                           src_raw, pred_sents,
            #                           attn[b], pred_score[b], gold_sent,
            #                           gold_score[b])
            # src = self.spm.DecodeIds([int(t) for t in translation_batch['batch'].src[0][5] if int(t) != len(self.spm)])
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            score = pred_score[b]
            translation = (pred_sents, gold_sent, raw_src, score)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        score_path = self.args.result_path + '.%d.score' % step
        self.score_out_file = codecs.open(score_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            for batch in tqdm(data_iter):
                if self.args.recall_eval:
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src, score = trans
                    pred_str = pred.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace(
                        '[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]',
                                                                                                   '').strip()
                    gold_str = gold.strip()
                    if self.args.recall_eval:
                        _pred_str = ''
                        gap = 1e3
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str + '<q>' + sent.strip()
                            can_gap = math.fabs(len(_pred_str.split()) - len(gold_str.split()))
                            # if(can_gap>=gap):
                            if len(can_pred_str.split()) >= len(gold_str.split()) + 10:
                                pred_str = _pred_str
                                break
                            else:
                                gap = can_gap
                                _pred_str = can_pred_str

                        # pred_str = ' '.join(pred_str.split()[:len(gold_str.split())])
                    # self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                    # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    self.score_out_file.write(str(score[0].item()) + '\n')
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
                self.score_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()
        self.score_out_file.close()

        if step != -1:
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)
            return rouges
        return None

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src

        inactive = torch.zeros(batch_size * beam_size).cuda()
        raw_constraints = batch.constraints
        DEBUG = False
        # if any(raw_constraints):
        #     for ct, c in enumerate(raw_constraints):
        #         if len(c) != 0:
        #             print(ct, [tokenizer.convert_ids_to_tokens(p) for p in c])
        #     DEBUG = True
        # raw_constraints = [[] for _ in range(batch_size)]
        # raw_constraints[0] = [[2035, 4620, 2024, 9385, 2098]]  # 'copyrighted'
        # raw_constraints[1] = [[4620]]  # 4871-images 4620-'pictures'
        # raw_constraints[2] = [[2035, 4620, 2024, 9385, 2098]]  # 'pictures'
        # raw_constraints[3] = [[3145, 2824, 1024, 2]]  # ':'
        # raw_constraints[4] = [[3145, 2824, 1024]]  # ':'
        # raw_constraints = [[[2003, 2056, 2000, 2022]] for _ in range(batch_size)]  # 'is said to be'
        print_list = [1]

        constraints = init_constraints(raw_constraints, beam_size, tokenizer.vocab['[unused0]'],
                                       tokenizer.vocab['[unused1]'])
        # constraints = [ConstrainedHypothesis(rc, eos_id=tokenizer.vocab['[unused1]']) for rc in batch.raw_constraints]

        if self.args.decode_mode in ['ent', 'ent_gold', 'ent_gold_once']:
            # not sure how to better deal with different sizes. Speed is ok
            if self.args.decode_mode == 'ent':
                mlm_labels = batch.mlm_labels
            else:
                mlm_labels = batch.mlm_labels_gold
            entity_ids = []  # important token_ids in each sample
            for label in mlm_labels:
                entity_ids.extend([label[label != -100]] * beam_size)

        src_features = self.model.bert(src, segs, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = decoder_input.transpose(0, 1)

            # Decoder forward.
            dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                     step=step)
            # Generator forward.
            if self.args.decode_mode == 'normal':
                log_probs = self.generator(dec_out.transpose(0, 1).squeeze(0))
            else:
                # up-weighting entities
                probs = self.generator(dec_out.transpose(0, 1).squeeze(0))
                # # print(probs.mean(), probs.std(), probs.min(), probs.max())
                gain = probs.std() * 2
                # gain = probs.mean().abs() / 1
                for idx in range(probs.size(0)):
                    probs[idx][entity_ids[idx]] += gain
                log_probs = torch.nn.functional.log_softmax(probs, dim=-1)
                # print(log_probs[0, :5])

            vocab_size = log_probs.size(-1)
            if step < min_length:
                log_probs[:, self.end_token] = -1e20
            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(alive_seq.size(0)):
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##', '').split()
                        if len(words) <= 3:
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            curr_scores[i] = -10e20

            folded_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            # select top-k from beam * vocab
            topk_scores, topk_ids = folded_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty
            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)  # best_word_indices (batch_size, beam_size)
            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                        topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))  # (batch_size, beam_size)

            batch_index, topk_ids, topk_log_probs, constraints, inactive = dba_topk(batch_size=batch_index.size()[0],
                                                                                    beam_size=beam_size,
                                                                                    inactive=inactive,
                                                                                    scores=curr_scores * length_penalty,
                                                                                    hypotheses=constraints,
                                                                                    best_ids=batch_index,
                                                                                    best_word_ids=topk_ids,
                                                                                    seq_scores=topk_log_probs,
                                                                                    debug=False)
            select_indices = batch_index.view(-1)
            if self.args.decode_mode in ['ent_gold_once']:
                # remove ids from entity_ids once they appear in the beam
                for ct, id in enumerate(topk_ids.view(-1)):
                    entity_ids[ct] = entity_ids[ct][entity_ids[ct] != id]
                    # if tokenizer.convert_ids_to_tokens(id.item()) == 'uk':
                    #     s = tokenizer.convert_ids_to_tokens(entity_ids[ct].cpu().numpy())
                    #     print(s, tokenizer.convert_ids_to_tokens(id.item()))
                    #     s = tokenizer.convert_ids_to_tokens(alive_seq[ct].cpu().numpy())
                    #     print(f'current seq at {ct}', s)
                    #     tm = 1
                # update by selected indices
                entity_ids = [entity_ids[idx] for idx in select_indices.cpu().numpy()]

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)
            if DEBUG:
                for seq, score in zip(alive_seq[beam_size * 1:beam_size * 2].cpu().numpy(), topk_scores[1]):
                    print('alive_seq:', tokenizer.decode(seq), score.item())
                print()
            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                # print('debug: reach max_length!')
                is_finished.fill_(1)
            # End condition is top beam is finished (top beam best_hyp[0] is then used as prediction)
            end_condition = is_finished[:, 0].eq(1)
            # 1Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    # when top beam finishes, all pred are marked finished
                    if end_condition[i]:
                        if DEBUG and i in print_list:
                            print('top beam ends!!!')
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        if DEBUG and i in print_list:
                            print('finished:', tokenizer.decode(predictions[i, j, 1:].cpu().numpy()), (i, j.item()),
                                  topk_scores[i, j].item())
                        # already ensure hypotheses are either finished or (unfinished but) have lower score than top
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]
                        if DEBUG and i in print_list:
                            print('output:', tokenizer.decode(pred.cpu().numpy()), score.item(), '\n')
                            _ = 1
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # NB Remove finished samples for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))

                inactive = inactive.view(-1, beam_size).index_select(0, non_finished).view(-1)
                constraints_new = []
                for i in non_finished:
                    i = i.item()
                    constraints_new.extend(constraints[i * beam_size: i * beam_size + beam_size])
                constraints = constraints_new

            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
