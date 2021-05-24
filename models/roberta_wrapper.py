from models.mlm_wrapper import MLMWrapper
from transformers import RobertaTokenizer, RobertaForMaskedLM
from utils.utils import load_json_dic, store_json_dic
import os


def store_roberta_vocab2idx(tokenizer, vocab2idx_file):
    vocab = tokenizer.get_vocab()
    vocab2idx = {}
    for token in vocab:
        word = tokenizer.decode(vocab[token]).strip()
        vocab2idx[word] = vocab[token]
    store_json_dic(vocab2idx_file, vocab2idx)


class RobertaWrapper(MLMWrapper):
    def __init__(self,
                 tokenizer: RobertaTokenizer,
                 model: RobertaForMaskedLM,
                 vocab2idx_file="data/roberta_data/vocab2idx.json",
                 device: int = None):
        super().__init__(tokenizer, model, device=device)
        if not os.path.isfile(vocab2idx_file):
            store_roberta_vocab2idx(tokenizer, vocab2idx_file)
        self.vocab2idx = load_json_dic(vocab2idx_file)

    def token_to_idx(self, token):
        return self.vocab2idx[token]
