import pickle
from collections import defaultdict
from pathlib import Path

from gifflar.tokenize.pretokenize import PreTokenizer


class TokenizerTrainer:
    def __init__(
            self,
            pre_tokenizer: PreTokenizer,
            corpus_path: str | Path,
            token_path: str | Path,
            total_token: int | None | list[int] = None,
            add_token: int | None | list[int] = None,
            save_format: str | None = None
    ):
        self.pre_tokenizer = pre_tokenizer
        self.save_format = save_format or "{:03d}"

        with open(corpus_path, "r") as f:
            self.corpus = [line.replace("\n", "") for line in f.readlines()]

        self.vocab = self.init_vocab(token_path)

        if isinstance(total_token, int) or isinstance(add_token, int):
            self.vocab_size = total_token or len(self.vocab) + add_token
        else:
            self.vocab_steps = total_token or [len(self.vocab) + n for n in add_token]
            self.vocab_size = self.vocab_steps[-1]

        self.pair_freqs = defaultdict(int)
        self.merges = {}
        self.splits = {}

    def init_vocab(self, token_path):
        raise NotImplementedError()

    def merge_pair(self, a, b):
        merge_token = a + b[2:] if b[:2] == "##" else a + b
        new_freqs = defaultdict(int)

        for word in self.corpus:
            tokens = self.splits[word]

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
            self.splits[word] = tokens

        return new_freqs, merge_token

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.vocab, self.merges), f)
        print(f"\nSaved to {path}")


class BPETrainer(TokenizerTrainer):
    def __init__(
            self,
            pre_tokenizer,
            corpus_path: str | Path,
            token_path: str | Path,
            total_token: int | None | list[int] = None,
            add_token: int | None | list[int] = None,
            save_format: str | None = None
    ):
        super(BPETrainer, self).__init__(pre_tokenizer, corpus_path, token_path, total_token, add_token, save_format)
        for word in self.corpus():
            tokenization = self.pre_tokenizer(word)
            if tokenization is not None:
                self.splits[word] = tokenization

    def init_vocab(self, token_path):
        with open(token_path, "r") as f:
            vocab = [v.strip() for v in f.readlines()]
        return vocab

    def compute_pair_freqs(self):
        for word in self.corpus:
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                self.pair_freqs[pair] += 1

    def train(self):
        self.compute_pair_freqs()
        while len(self.vocab) < self.vocab_size:
            print(f"\rVocab-size: {len(self.vocab)} / {self.vocab_size}", end="")
            best_pair, max_freq = "", None
            for pair, freq in self.pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            self.pair_freqs, merge_token = self.merge_pair(*best_pair)
            self.merges[best_pair] = merge_token
            self.vocab.append(merge_token)
            if len(self.vocab) in self.vocab_steps:
                self.save(f"{self.save_format.format(len(self.vocab))}.pkl")


class WordpieceTrainer(TokenizerTrainer):
    def __init__(
            self,
            pre_tokenizer,
            corpus_path: str | Path,
            token_path: str | Path,
            total_token: int | None | list[int] = None,
            add_token: int | None | list[int] = None,
            save_format: str | None = None,
    ):
        super(WordpieceTrainer, self).__init__(pre_tokenizer, corpus_path, token_path, total_token, add_token, save_format)
        for word in self.corpus:
            # print(word)
            tokenization = self.pre_tokenizer(word)
            # print(tokenization)
            if tokenization is not None:
                self.splits[word] = [t if i == 0 else "##" + t for i, t in enumerate(self.pre_tokenizer(word))]

        self.single_freqs = defaultdict(int)

    def init_vocab(self, token_path):
        self.vocab = []
        with open(token_path, "r") as f:
            for v in f.readlines():
                v = v.strip()
                self.vocab += [v, f"##{v}"]
        return self.vocab

    def compute_pair_scores(self):
        for word in self.corpus:
            split = self.splits[word]
            if len(split) == 1:
                self.single_freqs[split[0]] += 1
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                self.single_freqs[split[i]] += 1
                self.pair_freqs[pair] += 1
            self.single_freqs[split[-1]] += 1

    def train(self):
        self.compute_pair_scores()
        while len(self.vocab) < self.vocab_size:
            print(f"\rVocab-size: {len(self.vocab)} / {self.vocab_size}", end="")
            scores = {
                pair: freq / (self.single_freqs[pair[0]] * self.single_freqs[pair[1]])
                for pair, freq in self.pair_freqs.items()
            }
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            freq = self.pair_freqs[best_pair]
            self.pair_freqs, merge_token = self.merge_pair(*best_pair)
            self.single_freqs[best_pair[0]] -= freq
            self.single_freqs[best_pair[1]] -= freq
            self.single_freqs[merge_token] = freq
            self.vocab.append(merge_token)
            if len(self.vocab) in self.vocab_steps:
                self.save(f"{self.save_format.format(len(self.vocab))}.pkl")
