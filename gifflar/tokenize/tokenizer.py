import pickle
from pathlib import Path
from typing import Literal

import transformers
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from gifflar.tokenize.pretokenize import PreTokenizer


class GIFFLARTokenizer(PreTrainedTokenizerFast):
    def __init__(self, pre_tokenizer: PreTokenizer, mode: Literal["BPE", "WP"], base_vocab: list[str] | Path | str | None = None, *args, **kwargs):
        super(GIFFLARTokenizer, self).__init__(tokenizer_object=Tokenizer(models.Model()), *args, **kwargs)
        self.pre_tokenizer_ = pre_tokenizer
        self.mode = mode
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.bos_token = "[BOS]"
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.eos_token = "[EOS]"
        self.vocab_ = {token: i for i, token in enumerate([self.pad_token, self.bos_token, self.unk_token, self.sep_token, self.mask_token, self.eos_token, self.cls_token])}
        if base_vocab is not None:
            if not isinstance(base_vocab, list):
                with open(base_vocab, "r") as f:
                    base_vocab = [v.strip() for v in f.readlines()]
            self.vocab_.update({v: i for i, v in enumerate(base_vocab)})
        self.merges_ = {}

    @property
    def vocab_size(self):
        return len(self.vocab_)

    @property
    def pad_token_id(self):
        # This has to be 0 for some reason deep down the ESM processing
        return 0

    @property
    def bos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return 2

    @property
    def sep_token_id(self):
        return 3

    @property
    def mask_token_id(self):
        return 4

    @property
    def eos_token_id(self):
        return 5

    @property
    def cls_token_id(self):
        return 6

    def __len__(self):
        return len(self.vocab_)

    def bpe_tokenize(self, text):
        splits = self.pre_tokenizer_(text)
        for pair, merge in self.merges.items():
            i = 0
            while i < len(splits) - 1:
                if splits[i] == pair[0] and splits[i + 1] == pair[1]:
                    splits = splits[:i] + [merge] + splits[i + 2:]
                i += 1
        return splits

    def wordpiece_tokenize(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab_:
                i -= 1
            if i == 0:
                tokens.append("[UNK]")
                word = word[3:]
                if len(word) == 0:
                    return tokens
                word = "##" + word
                continue
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def load(self, path):
        with open(path, "rb") as f:
            base_vocab, self.merges = pickle.load(f)
        self.vocab_.update({v: i + self.eos_token_id for i, v in enumerate(base_vocab)})

    def __call__(self, text, *args, **kwargs):
        tokens = self.bpe_tokenize(text) if self.mode == "BPE" else self.wordpiece_tokenize(text)
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for token in tokens:
            input_ids.append(self.vocab_[token])
            token_type_ids.append(0)
            attention_mask.append(1)

        return transformers.tokenization_utils_base.BatchEncoding({
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        })