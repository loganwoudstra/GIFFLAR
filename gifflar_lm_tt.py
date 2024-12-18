from pathlib import Path
from collections import defaultdict

from gifflar.grammar.GIFFLARLexer import GIFFLARLexer
from gifflar.grammar.GIFFLARParser import GIFFLARParser
from antlr4 import InputStream, CommonTokenStream

import transformers
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, EsmConfig, EsmForMaskedLM, Trainer, TrainingArguments
from tokenizers import Tokenizer, NormalizedString, PreTokenizedString, pre_tokenizers, Encoding
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer
from datasets import load_dataset
from tokenizers import models, normalizers, processors
import tokenizers
import random


MAX_LENGTH = 200
random.seed(42)


class TokenizerGIFFLAR(PreTrainedTokenizerFast):
    def __init__(self, base_vocab: list[str] | Path | str | None = None, *args, **kwargs):
        super(TokenizerGIFFLAR, self).__init__(tokenizer_object=Tokenizer(models.Model()), *args, **kwargs)
        self.cls_token = "[CLS]"
        self.bos_token = "[BOS]"
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self.vocab_ = {}
        if base_vocab is not None:
            if not isinstance(base_vocab, list):
                with open(base_vocab, "r") as f:
                    base_vocab = [v.strip() for v in f.readlines()]
            self.vocab_.update({v: i for i, v in enumerate(base_vocab)})
        self.mergers_ = {}

    @property
    def vocab_size(self):
        return len(self.vocab_)

    @property
    def cls_token_id(self):
        return len(self.vocab_)

    @property
    def bos_token_id(self):
        return len(self.vocab_) + 1

    @property
    def unk_token_id(self):
        return len(self.vocab_) + 2

    @property
    def sep_token_id(self):
        return len(self.vocab_) + 3

    @property
    def mask_token_id(self):
        return len(self.vocab_) + 4

    @property
    def eos_token_id(self):
        return len(self.vocab_) + 5

    @property
    def pas_token_id(self):
        return len(self.vocab_) + 6

    def __len__(self):
        return len(self.vocab_)

    def _gifflar_pre_tokenize(self, iupac):
        iuapc = iupac.strip().replace(" ", "")
        token = CommonTokenStream(GIFFLARLexer(InputStream(data="{" + iupac + "}")))
        GIFFLARParser(token).start()
        return [t.text for t in token.tokens[1:-2]]

    def _merge(self, tokens):
        pass

    def train_bpe(self, corpus):
        pass

    def __call__(self, text, *args, **kwargs):
        tokens = self._gifflar_pre_tokenize(text)
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


def bpe_compute_pair_freqs(splits, corpus):
    pair_freqs = defaultdict(int)
    for word in corpus:
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += 1
    return pair_freqs


def bpe_merge_pair_gpt(a, b, splits, corpus):
    merge_token = a + b[2:] if b[:2] == "##" else a + b
    new_freqs = defaultdict(int)

    for word in corpus:
        tokens = splits[word]

        for i in range(len(tokens)):
            if i < len(tokens) - 1:
                # Check if the current pair is (a, b)
                if tokens[i] == a and tokens[i + 1] == b:
                    # Merge the pair
                    tokens = tokens[:i] + [merge_token] + tokens[i + 2:]

            # Update frequencies on the go if there's at least one token in new_tokens
            if 0 < i <= len(tokens) - 1:
                new_freqs[(tokens[i - 1], tokens[i])] += 1

        # Update the tokenization for the current word
        splits[word] = tokens

    return splits, new_freqs, merge_token


def bpe_merge_pair(a, b, splits, corpus, freqs):
    merge = a + b[2:] if b[:2] == "##" else a + b
    del_freqs = set()
    del_freqs.add((a, b))
    for word in corpus:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [merge] + split[i + 2:]
                if i > 0:
                    freqs[(split[i - 1], merge)] += 1
                    freqs[(split[i - 1], a)] -= 1
                    if freqs[(split[i - 1], a)] <= 0:
                        del_freqs.add((split[i - 1], a))
                if i < len(split) - 1:
                    freqs[(merge, split[i + 1])] += 1
                    freqs[(b, split[i + 1])] -= 1
                    if freqs[(b, split[i + 1])] <= 0:
                        del_freqs.add((b, split[i + 1]))
            #if split[i] in a or (i < len(split) - 1 and split[i + 1] == b):
            #    freqs[(split[i], split[i + 1])] += 1
            i += 1
        splits[word] = split
    # print(del_freqs)
    for pair in del_freqs:
        if freqs[*pair] <= 0 or pair == (a, b):
            del freqs[*pair]
    return splits, freqs


def bpe(token_path="gifflar/grammar/tokens.txt", corpus_path="glycans_1000.txt", num_token: int = 50):
    with open(token_path, "r") as f:
        base_vocab = [v.strip() for v in f.readlines()]

    with open(corpus_path, "r") as f:
        corpus = [line.strip().replace(" ", "") for line in f.readlines()]

    splits = {word: tg._gifflar_pre_tokenize(word) for word in corpus}

    vocab_size = num_token + len(base_vocab)
    merges = {}

    pair_freqs = bpe_compute_pair_freqs(splits, corpus)
    while len(base_vocab) < vocab_size:
        best_pair, max_freq = "", None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits, pair_freqs, merge_token = bpe_merge_pair_gpt(*best_pair, splits, corpus)
        merges[best_pair] = merge_token
        base_vocab.append(merge_token)
    return base_vocab, merges


def bpe_tokenize(text, merges):
    splits = tg._gifflar_pre_tokenize(text)
    for pair, merge in merges.items():
        i = 0
        while i < len(splits) - 1:
            if splits[i] == pair[0] and splits[i + 1] == pair[1]:
                splits = splits[:i] + [merge] + splits[i + 2 :]
            else:
                i += 1
    return splits


tg = TokenizerGIFFLAR("gifflar/grammar/tokens.txt")
base_vocab_test, merges_test = bpe(corpus_path="data/pretrain/glycans_100.txt", num_token=50)
print(bpe_tokenize("NeuNAc(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-3)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
                   merges_test))