from pathlib import Path

from datasets import load_dataset
from tokenizers import Encoding
from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer, EsmForMaskedLM, EsmConfig

from gifflar.tokenize.pretokenize import GrammarPreTokenizer
from gifflar.tokenize.tokenizer import GIFFLARTokenizer


MAX_LENGTH = 200
HIDDEN_LAYERS = 6
HIDDEN_SIZE = 320
ATTENTION_HEADS = 16
EPOCHS = 3
BATCH_SIZE = 8
OUTPUT_DIR = Path("GlyLM/bpe_glyles_2500_model")  # reset to /scratch/
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bpe = GIFFLARTokenizer(GrammarPreTokenizer(), "BPE")
bpe.load("GlyLM/bpe_glyles_2500.pkl")

wp = GIFFLARTokenizer(GrammarPreTokenizer(), "WP")
wp.load("GlyLM/wordpiece_lib_5000.pkl")

tokenizer = bpe

def tokenize_function(entries: dict) -> Encoding:
    return tokenizer(entries["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

dataset = load_dataset("text", data_files={"train": "datasets/glycans_1000.txt"})
proc_ds = dataset.map(tokenize_function, batched=False, remove_columns=["text"])

esm_config = EsmConfig(
    vocab_size=tokenizer.vocab_size + 7,  # accounting for special tokens
    num_hidden_layers=HIDDEN_LAYERS,
    hidden_size=HIDDEN_SIZE,
    num_attention_heads=ATTENTION_HEADS,
    pad_token_id=tokenizer.pad_token_id,
    max_position_embeddings=tokenizer.vocab_size + MAX_LENGTH,
    position_embedding_type="absolute",  # try: "rotary"
)

model = EsmForMaskedLM(
    config=esm_config,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    # bf16=True,
    # use_cpu=True,
    # load_best_model_at_end=True,
    save_strategy="epoch",
    # eval_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=proc_ds["train"],
)

trainer.train()
trainer.save_model(OUTPUT_DIR / "checkpoint-last")
