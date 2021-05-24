import torch
from tqdm import tqdm
from utils.constant import CUDA_DEVICE, BATCH_SIZE
from utils.utils import batch_text, score_to_result, get_pair, model_prefix, load_json_dic
from transformers import AutoTokenizer, BertForMaskedLM
import copy
import numpy as np


def count_sub_words(token, tokenizer):
    text_ids = tokenizer.encode(token, add_special_tokens=False)
    return len(text_ids)


def count_roberta_sub_words(template, obj, tokenizer):
    sub_exclude = template.replace("[Y]", "")
    sub_include = template.replace("[Y]", obj)
    sub_exclude = sub_exclude.replace("  ", " ")
    sub_include = sub_include.replace("  ", " ")
    sub_exclude_len = count_sub_words(sub_exclude, tokenizer)
    sub_include_len = count_sub_words(sub_include, tokenizer)
    return sub_include_len - sub_exclude_len


def template_to_sent(template, sub, obj, tokenizer):
    sent = template.replace('[X]', sub)
    mask_token = tokenizer.mask_token
    if "roberta" in tokenizer.name_or_path:
        mask_cnt = count_roberta_sub_words(sent, obj, tokenizer)
    else:
        mask_cnt = count_sub_words(obj, tokenizer)
    mask_obj = " ".join([mask_token] * mask_cnt)
    sent = sent.replace('[Y]', mask_obj)
    return sent


def get_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value == item]


def get_predict_score_with_multi_mask(tokenizer, model, input_text, args=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE

    mask_mark = tokenizer.mask_token
    mask_id = tokenizer.convert_tokens_to_ids(mask_mark)

    inputs = tokenizer.batch_encode_plus(
        input_text, padding='longest', truncation=True, max_length=256
    )

    input_ids = inputs['input_ids']
    mask_index = []
    for ids in input_ids:
        index = get_index(ids, mask_id)
        mask_index.append(index)

    for key in inputs:
        inputs[key] = torch.tensor(inputs[key]).cuda(cuda_device)
    outputs = model(**inputs)
    prediction_score = outputs[0]
    return prediction_score, mask_index


def score_to_id(score):
    score = torch.softmax(score, dim=-1)
    predict_id = torch.argmax(score).item()
    return predict_id


def greedy_search(predict_score, mask_index, tokenizer):
    ids = []
    for index in mask_index:
        score = predict_score[index]
        predict_id = score_to_id(score)
        ids.append(predict_id)
    predict_token = tokenizer.decode(ids)
    return predict_token


def mlm_predict_with_multi_mask(tokenizer, model,
                                input_texts, batch_size=BATCH_SIZE,
                                decode_method='greedy', beam_size=100,
                                args=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE
    batch_input_text = batch_text(input_texts, batch_size=batch_size)

    model = model.cuda(cuda_device)
    model.eval()
    predict_results = []
    for idx in tqdm(range(len(batch_input_text))):
        input_text = batch_input_text[idx]
        prediction_score, mask_index = get_predict_score_with_multi_mask(tokenizer, model, input_text, args=args)
        for i in range(len(mask_index)):
            if decode_method == 'greedy':
                predict_token = greedy_search(prediction_score[i], mask_index[i], tokenizer)
            else:
                raise RuntimeError("no decode method")
            predict_results.append(predict_token)
    return predict_results
