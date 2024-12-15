from datasets import load_dataset
from tokenizers import Encoding
from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer, EsmForMaskedLM, EsmConfig

from gifflar.tokenize.pretokenize import GrammarPreTokenizer
from gifflar.tokenize.tokenizer import GIFFLARTokenizer

bpe = GIFFLARTokenizer(GrammarPreTokenizer())
bpe.load("datasets/bpe_glyles_50.pkl")

wp = GIFFLARTokenizer(GrammarPreTokenizer())
wp.load("datasets/wp_glycowork_50.pkl")

t = wp
MAX_LENGTH = 200

def tokenize_function(entries: dict) -> Encoding:
    return t(entries["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

dataset = load_dataset("text", data_files={"train": "datasets/glycans_1000.txt"})
proc_ds = dataset.map(tokenize_function, batched=False, remove_columns=["text"])

esm_config = EsmConfig(
    vocab_size=t.vocab_size + 7,  # accounting for special tokens
    num_hidden_layers=6,
    hidden_size=320,
    num_attention_heads=16,
    pad_token_id=t.pad_token_id,
    max_position_embeddings=t.vocab_size + MAX_LENGTH,
    position_embedding_type="absolute",  # try: "rotary"
)

model = EsmForMaskedLM(
    config=esm_config,
).to("cpu")

data_collator = DataCollatorForLanguageModeling(tokenizer=t, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="GlyLM",
    num_train_epochs=2,
    per_device_train_batch_size=6,
    # bf16=True,
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=t,
    train_dataset=proc_ds["train"],
)

trainer.train()