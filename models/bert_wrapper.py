from models.mlm_wrapper import MLMWrapper
from transformers import BertForMaskedLM, BertTokenizer


class BertWrapper(MLMWrapper):
    def __init__(self, tokenizer: BertTokenizer, model: BertForMaskedLM, device: int = None):
        super().__init__(tokenizer, model, device=device)
