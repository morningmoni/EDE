import copy
from operator import attrgetter
from typing import List, Optional, Set

import torch

RawConstraintList = List[List[int]]

# NB. for debugging only
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')


class ConstrainedHypothesis:
    """
    Represents a set of words and phrases that must appear in the output.
    A constraint is of two types: sequence or non-sequence.
    A non-sequence constraint is a single word and can therefore be followed by anything,
    whereas a sequence constraint must be followed by a particular word (the next word in the sequence).
    This class also records which constraints have been met.
    A list of raw constraints is maintained internally as two parallel arrays. The following raw constraint
    represents two phrases that must appear in the output: 14 and 19 35 14.
        raw constraint: [[14], [19, 35, 14]]
    This is represented internally as:
        constraints: [14 19 35 14]
        is_sequence: [False False True True]
    That is, the constraints are simply concatenated, and we maintain a parallel array indicating whether each
    token ID must be followed by the next token ID. The same token ID can be present any number of times.
    :param constraint_list: A list of zero or raw constraints (each represented as a list of integers).
    :param eos_id: The end-of-sentence ID.
    """

    def __init__(self,
                 constraint_list: RawConstraintList,
                 eos_id: int) -> None:

        # `constraints` records the words of the constraints, as a list (duplicates allowed).
        # `is_sequence` is a parallel array that records, for each corresponding constraint,
        #    whether the current word is the non-final word of a phrasal constraint.
        self.constraints = []  # type: List[int]
        self.is_sequence = []  # type: List[bool]
        for phrase in constraint_list:
            self.constraints += phrase
            self.is_sequence += [True] * len(phrase)
            self.is_sequence[-1] = False

        self.eos_id = eos_id

        # no constraints have been met
        self.met = [False for x in self.constraints]
        self.last_met = -1

    def __len__(self) -> int:
        """
        :return: The number of constraints.
        """
        return len(self.constraints)

    def __str__(self) -> str:
        s = []
        for i, word_id in enumerate(self.constraints):
            s.append(str(word_id) if self.met[i] is False else 'X')
            if self.is_sequence[i]:
                s.append('->')
        return ' '.join(s)

    def size(self) -> int:
        """
        :return: the number of constraints
        """
        return len(self.constraints)

    def num_met(self) -> int:
        """
        :return: the number of constraints that have been met.
        """
        return sum(self.met)

    def num_needed(self) -> int:
        """
        :return: the number of un-met constraints.
        """
        return self.size() - self.num_met()

    def allowed(self) -> Set[int]:
        """
        Returns the set of constrained words that could follow this one.
        For unfinished phrasal constraints, it is the next word in the phrase.
        In other cases, it is the list of all unmet constraints.
        If all constraints are met, an empty set is returned.
        :return: The ID of the next required word, or -1 if any word can follow
        """
        items = set()  # type: Set[int]
        # Add extensions of a started-but-incomplete sequential constraint
        if self.last_met != -1 and self.is_sequence[self.last_met] == 1:
            word_id = self.constraints[self.last_met + 1]
            if word_id != self.eos_id or self.num_needed() == 1:
                items.add(word_id)

        # Add all constraints that aren't non-initial sequences
        else:
            for i, word_id in enumerate(self.constraints):
                if not self.met[i] and (i == 0 or not self.is_sequence[i - 1]):
                    if word_id != self.eos_id or self.num_needed() == 1:
                        items.add(word_id)

        return items

    def finished(self) -> bool:
        """
        Return true if all the constraints have been met.
        :return: True if all the constraints are met.
        """
        return self.num_needed() == 0

    def satisfy_k(self, k) -> bool:
        """
        :return: True if at least k constraints are met.
        """
        return self.num_met() >= k

    def is_valid(self, wordid, k=None) -> bool:
        """
        Ensures </s> is only generated when the hypothesis is completed.
        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        if k is None:
            # set to full constraint
            k = self.size()
        return self.satisfy_k(k) or wordid != self.eos_id or (self.num_needed() == 1 and self.eos_id in self.allowed())
        # return self.finished() or wordid != self.eos_id or (self.num_needed() == 1 and self.eos_id in self.allowed())

    def is_valid_full(self, wordid) -> bool:
        """
        Ensures </s> is only generated when the hypothesis is completed.
        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        return self.finished() or wordid != self.eos_id or (self.num_needed() == 1 and self.eos_id in self.allowed())

    def advance(self, word_id: int) -> 'ConstrainedHypothesis':
        """
        Updates the constraints object based on advancing on word_id.
        There is a complication, in that we may have started but not
        yet completed a multi-word constraint.  We need to allow constraints
        to be added as unconstrained words, so if the next word is
        invalid, we must "back out" of the current (incomplete) phrase,
        re-setting all of its words as unmet.
        :param word_id: The word ID to advance on.
        :return: A deep copy of the object, advanced on word_id.
        """

        obj = copy.deepcopy(self)

        # First, check if we're updating a sequential constraint.
        if obj.last_met != -1 and obj.is_sequence[obj.last_met] == 1:
            if word_id == obj.constraints[obj.last_met + 1]:
                # Here, the word matches what we expect next in the constraint, so we update everything
                obj.met[obj.last_met + 1] = True
                obj.last_met += 1
            else:
                # Here, the word is not the expected next word of the constraint, so we back out of the constraint.
                index = obj.last_met
                while obj.is_sequence[index]:
                    obj.met[index] = False
                    index -= 1
                obj.last_met = -1

        # If not, check whether we're meeting a single-word constraint
        else:
            # Build a list from all constraints of tuples of the
            # form (constraint, whether it's a non-initial sequential, whether it's been met)
            constraint_tuples = list(zip(obj.constraints, [False] + obj.is_sequence[:-1], obj.met))
            # We are searching for an unmet constraint (word_id) that is not the middle of a phrase and is not met
            query = (word_id, False, False)
            try:
                pos = constraint_tuples.index(query)
                obj.met[pos] = True
                obj.last_met = pos
            except ValueError:
                # query not found; identical but duplicated object will be returned
                pass

        return obj


class ConstrainedCandidate:
    """
    Object used to hold candidates for the beam in topk().
    :param row: The row in the scores matrix.
    :param col: The column (word ID) in the scores matrix.
    :param score: the associated accumulated score.
    :param hypothesis: The ConstrainedHypothesis containing information about met constraints.
    """

    __slots__ = ('row', 'col', 'score', 'hypothesis')

    def __init__(self,
                 row: int,
                 col: int,
                 score: float,
                 hypothesis: ConstrainedHypothesis) -> None:
        self.row = row
        self.col = col
        self.score = score
        self.hypothesis = hypothesis

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.hypothesis.num_met())


def init_constraints(raw_constraints: List[Optional[RawConstraintList]],
                     beam_size: int,
                     start_id: int,
                     eos_id: int) -> List[Optional[ConstrainedHypothesis]]:
    """
    duplicate the constraints x beam_size times
    :param raw_constraints: The list of raw constraints (list of list of IDs).
    :param beam_size: The beam size.
    :param start_id: The target-language vocabulary ID of the SOS symbol.
    :param eos_id: The target-language vocabulary ID of the EOS symbol.
    :return: A list of ConstrainedHypothesis objects (shape: (batch_size * beam_size,)).
    """
    constraints = [None] * (len(raw_constraints) * beam_size)  # type: List[Optional[ConstrainedHypothesis]]
    if any(raw_constraints):
        for i, raw_list in enumerate(raw_constraints):
            num_constraints = sum([len(phrase) for phrase in raw_list]) if raw_list is not None else 0
            if num_constraints > 0:
                hyp = ConstrainedHypothesis(raw_list, eos_id)
                idx = i * beam_size
                constraints[idx:idx + beam_size] = [hyp.advance(start_id) for _ in range(beam_size)]

    return constraints


def get_bank_sizes(num_constraints: int,
                   beam_size: int,
                   candidate_counts: List[int]) -> List[int]:
    """
    Evenly distributes the beam across the banks, where each bank is a portion of the beam devoted
    to hypotheses having met the same number of constraints, 0..num_constraints.
    After the assignment, banks with more slots than candidates are adjusted.
    :param num_constraints: The number of constraints.
    :param beam_size: The beam size.
    :param candidate_counts: The empirical counts of number of candidates in each bank.
    :return: A distribution over banks.
    """

    num_banks = num_constraints + 1
    bank_size = beam_size // num_banks
    remainder = beam_size - bank_size * num_banks

    # Distribute any remainder to the end
    assigned = [bank_size for x in range(num_banks)]
    assigned[-1] += remainder

    # Now, moving right to left, push extra allocation to earlier buckets.
    # This encodes a bias for higher buckets, but if no candidates are found, space
    # will be made in lower buckets. This may not be the best strategy, but it is important
    # that you start pushing from the bucket that is assigned the remainder, for cases where
    # num_constraints >= beam_size.
    for i in reversed(range(num_banks)):
        overfill = assigned[i] - candidate_counts[i]
        if overfill > 0:
            assigned[i] -= overfill
            assigned[(i - 1) % num_banks] += overfill

    return assigned


def dba_topk(batch_size, beam_size, inactive, scores, hypotheses, best_ids, best_word_ids, seq_scores, debug=False,
             skip=False, partial=False, top_k=None, top_p=None, min_score=None, attn_scores=None, score_mode=0):
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.
    :param partial: whether allow partial (not all constraints are satisifed)
    :param top_k: if partial=True, take only top_k from vocab prob
    :param top_p: if partial=True, take only top_p from vocab prob
    :param min_score: if partial=True, take only > min_score from vocab prob
    :param score_mode: 0 use softmax vocab_prob, 1 use softmax copy_prob, 2/3 use self-attention scores (static)
    :param attn_scores: mode 0/1 have range [0, 1], 2/3 have range [0, src_len)
    :param batch_size: The number of segments (samples) in the batch.
    :param beam_size: The length of the beam for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size x batch_size)).
            if remaining hypos < beam_size, make rest inactive (works like padding), doesn't seem very
            useful in my experients since they are usually full, you can double-check
    :param scores: The scores array (shape: (beam_size x batch_size, target_vocab_size)).
            i change to the higher the better torch.argmax(scores, dim=1)
    :param hypotheses: The list of hypothesis objects (shape: (beam_size x batch_size)).
            where the constraints are stored
    :param best_ids: The current list of best hypotheses (shape: (batch_size, beam_size)).
            which beam of beam_size the best hypothesis is on
    :param best_word_ids: The parallel list of best word IDs (shape: (batch_size, beam_size)).
            which token_id for that beam
    :param seq_scores: (shape: (batch_size, beam_size)).
            the accumulated log probs so far
    :param debug: True to print info for debugging
    :param skip: True to skip DBA
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """
    # NB. this implementation uses for loop which may be slow
    # a batch version is at https://github.com/awslabs/sockeye/compare/master...edwardjhu:trie_constraints
    if hypotheses is None or skip:
        return best_ids, best_word_ids, seq_scores, hypotheses, inactive
    for sentno in range(batch_size):
        rows = slice(sentno * beam_size, sentno * beam_size + beam_size)
        if hypotheses[rows.start] is not None and hypotheses[rows.start].size() > 0:
            # best_ids[rows], best_word_ids[rows], seq_scores[rows], \
            # hypotheses[rows], inactive[rows] = _sequential_topk(beam_size,
            #                                                     inactive[rows],
            #                                                     scores[rows],
            #                                                     hypotheses[rows],
            #                                                     best_ids[rows] - rows.start,
            #                                                     best_word_ids[rows],
            #                                                     seq_scores[rows],
            #                                                     debug)

            best_ids[sentno], best_word_ids[sentno], seq_scores[sentno], \
            hypotheses[rows], inactive[rows] = _sequential_topk(beam_size,
                                                                inactive[rows],
                                                                scores[rows],
                                                                hypotheses[rows],
                                                                best_ids[sentno] - rows.start,
                                                                best_word_ids[sentno],
                                                                seq_scores[sentno],
                                                                debug,
                                                                partial, top_k, top_p, min_score,
                                                                (s[rows] for s in attn_scores) if attn_scores else None,
                                                                score_mode)

            # offsetting since the returned smallest_k() indices were slice-relative
            best_ids[sentno] += rows.start
        else:
            # If there are no constraints for this sentence in the batch, everything stays
            # the same, except we need to mark all hypotheses as active
            inactive[rows] = 0

    return best_ids, best_word_ids, seq_scores, hypotheses, inactive


def constraint_pass(idx, score_l, sorted_score_l, indices, min_score=None, top_k=None, top_p=None):
    if min_score is not None and score_l[idx] < min_score:
        return False
    loc = None
    if top_k is not None:
        assert top_k > 0 and type(top_k) == int
        loc = (indices == idx).nonzero()[0].item()
        if loc >= top_k:
            return False
    if top_p is not None:
        assert 0 <= top_p <= 1
        if loc is None:
            loc = (indices == idx).nonzero()[0].item()
        p = torch.sum(sorted_score_l[:loc])
        if p >= top_p:
            return False

    return True


def _sequential_topk(beam_size,
                     inactive,
                     scores,
                     hypotheses,
                     best_ids,
                     best_word_ids,
                     sequence_scores,
                     debug=False, partial=False, top_k=None, top_p=None, min_score=None,
                     attn_scores=None, score_mode=0):
    num_constraints = hypotheses[0].size()
    importance_score = [torch.exp(scores)]  # probs
    if partial:
        # allow current hypo w. highest #satisfied constraints to select EOS
        max_n_satisfied = max(hypo.num_met() for hypo in hypotheses)
        # which score to use for filtering
        if attn_scores is not None:
            importance_score.extend(attn_scores)
        assert score_mode < len(importance_score)
        if score_mode in [2, 3] and top_p is not None:
            print(f'Do not use top_p for score_mode={score_mode}')
            raise NotImplementedError
    else:
        max_n_satisfied = None  # None: has to satisfy all constraints

    candidates = set()
    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    if debug:
        print(f'constraints: {tokenizer.decode(hypotheses[0].constraints)}')
        print(f'max #constraints satisfied: {max_n_satisfied} ')
    for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
        row = int(row.item())
        col = int(col.item())
        if hypotheses[row] is None:
            continue
        if partial:
            valid = hypotheses[row].is_valid(col, k=0)  # NB. use max_n_satisfied
        else:
            valid = hypotheses[row].is_valid_full(col)
        if valid:
            seq_score = float(seq_score.item())
            new_item = hypotheses[row].advance(col)
            # if debug:
            #     if col == 2:
            #         print(f'#met={hypotheses[row].num_met()}')
            #     print(f'{tokenizer.decode([col])} {(row, col)}, {scores[row, col].item():.3f} overall best & valid')
            cand = ConstrainedCandidate(row, col, seq_score, new_item)
            candidates.add(cand)

    # For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    # best_next = mx.nd.argmin(scores, axis=1)
    best_next = torch.argmax(scores, dim=1)  # argmax since we use log_probs as scores
    for row in range(beam_size):
        if inactive[row]:
            continue

        hyp = hypotheses[row]

        # (2) add all the constraints that could extend this
        nextones = hyp.allowed()

        # if debug:
        #     for col in nextones:
        #         print(f'{tokenizer.decode([col])} {(row, col)} {scores[row, col].item():.3f} extend')

        # (3) add the single-best item after this (if it's valid)
        col = int(best_next[row].item())
        if hyp.is_valid(col):
            # if debug:
            #     print(f'{tokenizer.decode([col])} {(row, col)} {scores[row, col].item():.3f} best-next')
            nextones.add(col)
        if partial:
            score_l = importance_score[score_mode][row]
            # calc them only when in use
            if top_k is not None or top_p is not None:
                sorted_score_l, indices = torch.sort(score_l, descending=True)
            else:
                sorted_score_l, indices = None, None
        # Now, create new candidates for each of these items
        for col in nextones:
            new_item = hyp.advance(col)
            score = scores[row, col].item()
            # 1. at the 1st step other beams are -inf  2. when < MIN_LEN, EOS is -1e20
            # only add if its score is high enough
            if not partial or constraint_pass(col, score_l, sorted_score_l, indices, min_score, top_k, top_p):
                cand = ConstrainedCandidate(row, col, score, new_item)
                candidates.add(cand)
                if debug:
                    print(f'{tokenizer.decode([col])} {(row, col)} {scores[row, col].item():.3f} added!')
            elif debug:
                print(f'{tokenizer.decode([col])} {(row, col)} {scores[row, col].item():.3f} discarded!')

    # Sort the candidates. After allocating the beam across the banks, we will pick the top items
    # for each bank from this list
    sorted_candidates = sorted(candidates, key=attrgetter('score'), reverse=True)

    # The number of hypotheses in each bank
    counts = [0 for _ in range(num_constraints + 1)]
    for cand in sorted_candidates:
        counts[cand.hypothesis.num_met()] += 1

    # Adjust allocated bank sizes if there are too few candidates in any of them
    bank_sizes = get_bank_sizes(num_constraints, beam_size, counts)
    if debug:
        print('bank counts', counts)
        print('bank sizes', bank_sizes)

    # Sort the candidates into the allocated banks
    pruned_candidates = []  # type: List[ConstrainedCandidate]
    for i, cand in enumerate(sorted_candidates):
        bank = cand.hypothesis.num_met()

        if bank_sizes[bank] > 0:
            pruned_candidates.append(cand)
            bank_sizes[bank] -= 1

    num_pruned_candidates = len(pruned_candidates)  # pruned_candidates is the remaining cand in the whole beam

    inactive[:num_pruned_candidates] = 0

    # Pad the beam if there are unused slots so array assignment still works
    if num_pruned_candidates < beam_size:
        inactive[num_pruned_candidates:] = 1
        pruned_candidates += [pruned_candidates[num_pruned_candidates - 1]] * (beam_size - num_pruned_candidates)

    return (torch.LongTensor([x.row for x in pruned_candidates]),
            torch.LongTensor([x.col for x in pruned_candidates]),
            torch.FloatTensor([x.score for x in pruned_candidates]),
            [x.hypothesis for x in pruned_candidates],
            inactive)
