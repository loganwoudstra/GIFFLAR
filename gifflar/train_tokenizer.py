from gifflar.tokenize.pretokenize import GrammarPreTokenizer, GlycoworkPreTokenizer
from gifflar.tokenize.trainer import WordpieceTrainer, BPETrainer


trainers = {"wordpiece": WordpieceTrainer, "bpe": BPETrainer}
# for tokens, pt in [("glyles", GrammarPreTokenizer), ("lib", GlycoworkPreTokenizer)]:
for tokens, pt in [("glyles", GrammarPreTokenizer)]:
    # for tokens, pt in [("lib", GlycoworkPreTokenizer)]:
    # for name, trainer in [("wordpiece", WordpieceTrainer), ("bpe", BPETrainer)]:
    for name, trainer in [("bpe", BPETrainer)]:
        print(f"Training {name} tokenizer for {tokens}")
        tt = trainer(
            pt(),
            corpus_path=f"/home/daniel/Data1/roman/GIFFLAR/unique/glycoverse.txt",
            # corpus_path=f"/home/daniel/Desktop/GIFFLAR/datasets/glycans_1000.txt",
            token_path=f"datasets/{tokens}.txt",
            total_token=[500, 1_000],  # , 2_000, 5_000, 10_000],
            save_format=f"GlyLM_full/{name}_{tokens}_{{}}.pkl",
        )
        tt.train()
