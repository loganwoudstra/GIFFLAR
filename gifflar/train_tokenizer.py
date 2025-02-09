import sys

from gifflar.tokenize.pretokenize import GrammarPreTokenizer, GlycoworkPreTokenizer
from gifflar.tokenize.trainer import WordpieceTrainer, BPETrainer


PRE_TOKENIZERS = {"glyles": GrammarPreTokenizer, "lib": GlycoworkPreTokenizer}
TRAINER = {"wordpiece": WordpieceTrainer, "bpe": BPETrainer}

def train_tokenizer(name_trainer, name_pre_tokenizer):
    print(f"Training {name_pre_tokenizer} tokenizer with {name_trainer}")
    TRAINER[name_trainer](
        PRE_TOKENIZERS[name_pre_tokenizer](),
        corpus_path=f"/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/unique/unique_iupacs_137000.pkl",
        # corpus_path="/home/rjo21/Desktop/GIFFLAR/datasets/glycans_1000.txt",
        token_path=f"datasets/{name_pre_tokenizer}.txt",
        total_token=[2_500, 5_000, 7_500, 10_000],
        save_format=f"/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/unique/GlyLM/{name_trainer}_{name_pre_tokenizer}_{{}}.pkl",
    ).train()


def train_all():
    for name_trainer in ["wordpiece", "bpe"]:
        for name_pre_tokenizer in ["lib"]:  # ["glyles", "lib"]:
            train_tokenizer(name_trainer, name_pre_tokenizer)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_all()
    elif len(sys.argv) == 3 and sys.argv[1] in TRAINER and sys.argv[2] in PRE_TOKENIZERS:
        train_tokenizer(sys.argv[1], sys.argv[2])
    else:
        print(
            "Usage:\n"
            "------\n"
            "Either without arguments to train all combinations of trainers and tokenizers:\n"
            "\tpython -m gifflar.train_tokenizer\n"
            "or with specification of the trainer and pre-tokenizer to use (both must be specified):\n"
            "\tpython -m gifflar.train_tokenizer [wordpiece|bpe] [glyles|lib]"
        )

