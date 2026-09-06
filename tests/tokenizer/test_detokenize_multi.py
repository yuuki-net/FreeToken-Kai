"""The detokenizer must decode several messages for one uid in a batch (an MTP verify window
commits a whole draft window per step) exactly like the same tokens arriving one per batch."""
from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager

VOCAB = {1: "I", 2: " need", 3: " the", 4: " current", 5: " date", 6: ".", 7: "\n"}


class WordTokenizer:
    eos_token_id = 0

    def batch_decode(self, ids_list):
        return ["".join(VOCAB[i] for i in ids) for ids in ids_list]


def _msg(uid, tok, finished=False):
    return DetokenizeMsg(uid=uid, next_token=tok, finished=finished)


def _run(batches):
    man = DetokenizeManager(WordTokenizer())
    return "".join("".join(man.detokenize(b)) for b in batches)


def test_window_of_tokens_matches_one_per_step():
    toks = [1, 2, 3, 4, 5, 6, 7]
    one_per_step = _run([[_msg(7, t)] for t in toks])
    windows = _run([[_msg(7, 1)], [_msg(7, 2), _msg(7, 3), _msg(7, 4)], [_msg(7, 5), _msg(7, 6)], [_msg(7, 7)]])
    assert one_per_step == windows == "I need the current date.\n"


def test_two_uids_interleaved_in_one_batch():
    man = DetokenizeManager(WordTokenizer())
    out = man.detokenize([_msg(1, 1), _msg(2, 3), _msg(1, 2), _msg(1, 7), _msg(2, 7)])
    # per-message outputs stay in message order; each uid's text is contiguous and unduplicated
    assert "".join(o for o, m in zip(out, [1, 2, 1, 1, 2]) if m == 1) == "I need\n"
    assert "".join(o for o, m in zip(out, [1, 2, 1, 1, 2]) if m == 2) == " the\n"
