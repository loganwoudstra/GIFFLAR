import torch
from transformers import EsmModel

from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.downstream import DownstreamGGIN
from gifflar.model.utils import get_prediction_head
from gifflar.tokenize.pretokenize import GlycoworkPreTokenizer, GrammarPreTokenizer
from gifflar.tokenize.tokenizer import GIFFLARTokenizer

def pipeline(tokenizer, glycan_lm, iupac):
    try:
        tokens = tokenizer(iupac)
        input_ids = torch.tensor(tokens["input_ids"]).unsqueeze(0).to(glycan_lm.device)
        attention = torch.tensor(tokens["attention_mask"]).unsqueeze(0).to(glycan_lm.device)
        with torch.no_grad():
            return glycan_lm(input_ids, attention).last_hidden_state
    except:
        return None


class GlycanLM(DownstreamGGIN):
    def __init__(self, token_file, model_dir, hidden_dim: int, *args, **kwargs):
        super().__init__(feat_dim=1, hidden_dim=1, *args, **kwargs)
        del self.convs

        pretokenizer = GrammarPreTokenizer() if "glyles" in token_file else GlycoworkPreTokenizer()
        mode = "BPE" if "bpe" in token_file else "WP"
        tokenizer = GIFFLARTokenizer(pretokenizer, mode)
        tokenizer.load(token_file)
        glycan_lm = EsmModel.from_pretrained(model_dir)
        self.encoder = lambda x: pipeline(tokenizer, glycan_lm, x)
        self.head, self.loss, self.metrics = get_prediction_head(hidden_dim, self.output_dim, self.task)

    def forward(self, batch: HeteroDataBatch) -> dict[str, torch.Tensor]:
        """
        Make predictions based on the molecular fingerprint.

        Args:
            batch: Batch of heterogeneous graphs.

        Returns:
            Dict holding the node embeddings (None for the MLP), the graph embedding, and the final model prediction
        """
        with torch.no_grad():
            graph_embeddings = torch.cat([self.encoder(iupac).mean(dim=1) for iupac in batch["IUPAC"]], dim=0).to(self.device)
        return {
            "node_embed": None,
            "graph_embed": graph_embeddings,
            "preds": self.head(graph_embeddings),
        }
