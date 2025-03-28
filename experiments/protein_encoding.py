import re

import torch
from transformers import T5EncoderModel, AutoTokenizer, AutoModel, BertTokenizer, T5Tokenizer


class PLMEncoder:
    def __init__(self, layer_num: int):
        self.layer_num = layer_num
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, seq: str) -> torch.Tensor:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Ankh(PLMEncoder):
    # 450M parameters
    def __init__(self, layer_num: int):
        super().__init__(layer_num)
        self.tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
        self.model = T5EncoderModel.from_pretrained("ElnaggarLab/ankh-base").to(self.device)

    def forward(self, seq: str) -> torch.Tensor:
        outputs = self.tokenizer.batch_encode_plus(
            [list(seq)],
            add_special_tokens=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            ankh = self.model(
                input_ids=outputs["input_ids"].to(self.device),
                attention_mask=outputs["attention_mask"].to(self.device),
                output_attentions=False,
                output_hidden_states=True,
            )
        return ankh.hidden_states[self.layer_num][:, 1:].mean(dim=1)[0]


class ESM(PLMEncoder):
    # 650M parameters
    def __init__(self, layer_num: int):
        super().__init__(layer_num)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(self.device)

    def forward(self, seq: str) -> torch.Tensor:
        a = self.tokenizer(seq)
        with torch.no_grad():
            esm = self.model(
                torch.Tensor(a["input_ids"]).long().reshape(1, -1).to(self.device),
                torch.Tensor(a["attention_mask"]).long().reshape(1, -1).to(self.device),
                output_attentions=False,
                output_hidden_states=True,
            )
        return esm.hidden_states[self.layer_num][:, 1:-1].mean(dim=1)[0]


class ProtBERT(PLMEncoder):
    # 420M parameters ProtTrans paper
    def __init__(self, layer_num: int):
        super().__init__(layer_num)
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert").to(self.device)

    def forward(self, seq: str) -> torch.Tensor:
        sequence_w_spaces = ' '.join(seq)
        encoded_input = self.tokenizer(
            sequence_w_spaces,
            return_tensors='pt',
        )
        with torch.no_grad():
            protbert = self.model(
                input_ids=encoded_input["input_ids"].to(self.device),
                attention_mask=encoded_input["attention_mask"].to(self.device),
                token_type_ids=encoded_input["token_type_ids"].to(self.device),
                output_attentions=False,
                output_hidden_states=True,
            )
        return protbert.hidden_states[self.layer_num][:, 1:-1].mean(dim=1)[0]


class ProstT5(PLMEncoder):
    #
    def __init__(self, layer_num: int):
        super().__init__(layer_num)
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5")
        self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(self.device)

    def forward(self, seq: str) -> torch.Tensor:
        seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in [seq]]
        seq = ["<AA2fold>" + " " + s.upper() for s in seq]
        ids = self.tokenizer.batch_encode_plus(
            seq,
            add_special_tokens=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            prostt5 = self.model(
                ids.input_ids.to(self.device),
                attention_mask=ids.attention_mask.to(self.device),
                output_attentions=False,
                output_hidden_states=True,
            )
        return prostt5.hidden_states[self.layer_num][:, 1:-1].mean(dim=1)[0]


class AMPLIFY(PLMEncoder):
    def __init__(self, layer_num: int):
        super().__init__(layer_num)
        self.tokenizer = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True).to(self.device)

    def forward(self, seq: str) -> torch.Tensor:
        a = self.tokenizer.encode(seq, return_tensors="pt").to(self.device)
        with torch.no_grad():
            amplify = self.model(
                a,
                output_attentions=False,
                output_hidden_states=True,
            )
        pass


ENCODER_MAP = {
    "Ankh": Ankh,
    "ESM": ESM,
    "ProtBert": ProtBERT,
    "ProstT5": ProstT5,
    "AMPLIFY": AMPLIFY,
}

EMBED_SIZES = {
    "Ankh": 768,
    "ESM": 1280,
    "ProtBert": 1024,
    "ProstT5": 1024,
    "AMPLIFY": ...,
}

MAX_LAYERS = {
    "Ankh": 49,
    "ESM": 34,
    "ProtBert": 31,
    "ProstT5": 25,
    "AMPLIFY": ...,
}
