from utils.utils import MODEL_PATH, model_prefix
from transformers import AutoTokenizer, AutoModelForMaskedLM, \
    GPT2LMHeadModel, GPT2Tokenizer, \
    RobertaForMaskedLM, RobertaTokenizer, BertTokenizer, BertForMaskedLM, \
    BartForConditionalGeneration, BartTokenizer
from models.bert_wrapper import BertWrapper
from models.roberta_wrapper import RobertaWrapper


def build_model_wrapper(model_name, device=None):
    if model_name in MODEL_PATH:
        model_path = MODEL_PATH[model_name]
    else:
        raise RuntimeError('model not exsit')
    if model_prefix(model_name) == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForMaskedLM.from_pretrained(model_path)
        model_wrapper = RobertaWrapper(tokenizer, model, device=device)
    elif model_prefix(model_name) == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = BertForMaskedLM.from_pretrained(model_path)
        model_wrapper = BertWrapper(tokenizer, model, device=device)
    else:
        raise RuntimeError('model not exsit')
    return model_wrapper
