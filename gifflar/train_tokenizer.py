from gifflar.tokenize.pretokenize import GrammarPreTokenizer, GlycoworkPreTokenizer
from gifflar.tokenize.trainer import WordpieceTrainer, BPETrainer


trainers = {"wordpiece": WordpieceTrainer, "bpe": BPETrainer}
for tokens, pt in [("glyles", GrammarPreTokenizer), ("lib", GlycoworkPreTokenizer)]:
    for name, trainer in trainers.items():
        print(f"Training {name} tokenizer for {tokens}")
        tt = trainer(
            pt(),
            corpus_path=f"data/glycans_1000.txt",
            token_path=f"data/{tokens}.txt",
            total_token=[500, 1_000, 2_000, 5_000, 10_000],
            save_format=f"{name}_{tokens}_{{}}.pkl"
        )
        tt.train()
        # tt.save(f"data/{name}_{tokens}_50.pkl")
