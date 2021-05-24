from transformers import AutoTokenizer, AutoModelForMaskedLM, \
    GPT2LMHeadModel, GPT2Tokenizer, \
    RobertaForMaskedLM, RobertaTokenizer, BertTokenizer, BertForMaskedLM, \
    BartForConditionalGeneration, BartTokenizer
import torch
import json
from utils.constant import CUDA_DEVICE, RELATION_FILES
import numpy as np
import matplotlib.pyplot as plt
import random
from prettytable import PrettyTable
import seaborn as sns
import os
import pandas as pd
from matplotlib import rcParams

# You can put the model path here
MODEL_PATH = {
    'bert-large-cased': '/share/model/bert/cased_L-24_H-1024_A-16',
    'roberta-large': '/data/cbox/saved_models/roberta-large',
}


def build_model(model_name):
    if model_name in MODEL_PATH:
        model_path = MODEL_PATH[model_name]
    else:
        raise RuntimeError('model not exsit')
    if model_prefix(model_name) == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForMaskedLM.from_pretrained(model_path)
    elif model_prefix(model_name) == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = BertForMaskedLM.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    return tokenizer, model


def batch_text(input_texts, batch_size=32, obj_tokens=None):
    if obj_tokens is None:
        batch_input_text = []
        single_batch = []
        for text in input_texts:
            single_batch.append(text)
            if len(single_batch) == batch_size:
                batch_input_text.append(single_batch)
                single_batch = []
        if len(single_batch) > 0:
            batch_input_text.append(single_batch)
        return batch_input_text
    else:
        assert len(input_texts) == len(obj_tokens)
        batch_input_text = []
        batch_obj_tokens = []
        single_batch = []
        single_obj_batch = []
        for text, obj in zip(input_texts, obj_tokens):
            single_batch.append(text)
            single_obj_batch.append(obj)
            if len(single_batch) == batch_size:
                batch_input_text.append(single_batch)
                batch_obj_tokens.append(single_obj_batch)
                single_batch = []
                single_obj_batch = []
        if len(single_batch) > 0:
            batch_input_text.append(single_batch)
            batch_obj_tokens.append(single_obj_batch)
        return batch_input_text, batch_obj_tokens


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
        f.close()
    return data


def store_file(filename, data):
    with open(filename, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
    f.close()


def load_json_dic(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def store_json_dic(filename, dic):
    with open(filename, 'w') as f:
        json.dump(dic, f)


def load_roberta_vocab():
    return load_json_dic("data/roberta_data/roberta_vocab.json")


def model_prefix(model_name):
    return model_name.split("-")[0]


def filter_samples_by_vocab(samples, vocab):
    filter_samples = []
    for sample in samples:
        sub, obj = get_pair(sample)
        if obj in vocab:
            filter_samples.append(sample)
    return filter_samples, len(samples), len(filter_samples)


def get_relations(file_path='data/relations_with_trigger.jsonl'):
    original_relations = load_file(file_path)
    return original_relations


def score_to_result(score, topk, tokenizer, obj_token=None, rank_k=10000, roberta_vocab2idx=None):
    score = torch.softmax(score, dim=-1)
    predicted_prob, predicted_index = torch.topk(score, topk)
    predicted_prob = predicted_prob.detach().cpu().numpy()
    predicted_index = predicted_index.cpu().numpy().tolist()
    if "roberta" in tokenizer.name_or_path:
        predicted_tokens = []
        for index in predicted_index:
            predicted_tokens.append(tokenizer.decode(index).strip())
    elif 'bert' in tokenizer.name_or_path:
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_index)
    elif 'gpt' in tokenizer.name_or_path:
        predicted_tokens = []
        for index in predicted_index:
            predicted_tokens.append(tokenizer.decode(index))
    else:
        raise RuntimeError('model not defined')
    if obj_token is None:
        return predicted_tokens, predicted_prob
    else:
        if "roberta" in tokenizer.name_or_path:
            if roberta_vocab2idx is None:
                raise RuntimeError("need to be fix")
            obj_index = roberta_vocab2idx[obj_token]
            obj_prob = score[obj_index].item()
        else:
            obj_index = tokenizer.convert_tokens_to_ids(obj_token)
            obj_prob = score[obj_index].item()

        rank_prob, rank_index = torch.topk(score, rank_k)
        rank_index = rank_index.cpu().numpy().tolist()
        if obj_index not in rank_index:
            obj_rank = rank_k
            mrr = 0
        else:
            obj_rank = rank_index.index(obj_index) + 1
            mrr = 1 / obj_rank
        return predicted_tokens, predicted_prob, obj_prob, obj_rank, mrr


def get_pair(sample):
    while "sub_label" not in sample:
        sample = sample['sample']
    sub = sample['sub_label']
    obj = sample['obj_label']
    return sub, obj


def mean_round(num, num_len, r=2):
    return round(num * 100 / num_len, r)


def divide_samples_by_ans(samples):
    true_samples = []
    false_samples = []
    for sample in samples:
        if sample['predict_ans'] is True:
            true_samples.append(sample)
        else:
            false_samples.append(sample)
    return true_samples, false_samples


def box_plot(ax, data, labels=None):
    ax.boxplot(data, labels=labels)
    plt.show()


def get_relation_args(args):
    infos = RELATION_FILES[args.relation_type]
    args.relation_file = infos['relation_file']
    args.sample_dir = infos['sample_dir']
    args.sample_file_type = infos["sample_file_type"]
    return args


def get_bert_vocab(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    return vocab


def set_seed(seed_num=1023):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True


def get_table_stat(table: PrettyTable, return_cols=False):
    rows = table._rows
    mean_row = []
    median_row = []
    up_quantile_row = []
    down_quantile_row = []
    std_row = []

    cols = []

    if len(rows) == 0:
        return table

    for j in range(len(rows[0])):
        cols.append([row[j] for row in rows])

    for col in cols:
        if type(col[0]) == str:
            mean_row.append('mean')
            median_row.append('median')
            std_row.append('std')
            up_quantile_row.append('up_quantile')
            down_quantile_row.append('down_quantile')
        else:
            mean = round(float(np.mean(col)), 2)
            mean_row.append(mean)
            median = round(float(np.median(col)), 2)
            median_row.append(median)
            std = round(float(np.std(col)), 2)
            std_row.append(std)
            up_quantile = round(float(np.quantile(col, 0.25)), 2)
            up_quantile_row.append(up_quantile)
            down_quantile = round(float(np.quantile(col, 0.75)), 2)
            down_quantile_row.append(down_quantile)

    table.add_row(mean_row)
    table.add_row(up_quantile_row)
    table.add_row(median_row)
    table.add_row(down_quantile_row)
    table.add_row(std_row)
    if return_cols:
        return table, cols
    else:
        return table


def draw_heat_map(data, row_labels, col_labels,
                  pic_dir='pics/paper_pic/head_or_tail', pic_name='all_samples'):
    plt.figure(figsize=(8, 2))
    sns.set_theme()
    ax = sns.heatmap(data=data,
                     center=0,
                     annot=True, fmt='.2f',
                     xticklabels=row_labels,
                     yticklabels=col_labels)
    if not os.path.isdir(pic_dir):
        os.mkdir(pic_dir)
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig('{}/{}.eps'.format(pic_dir, pic_name), format='eps')
    plt.show()


def draw_box_plot(corrs, pic_name, pic_dir, ylim=None, hor=True):
    data = {"prompt": [], "corr": []}
    for prompt in corrs:
        for corr in corrs[prompt]:
            data["prompt"].append(prompt)
            data["corr"].append(corr)

    pd_data = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")

    if hor is True:
        flatui = ["#d6ecfa"]
        ax = sns.boxplot(
            x="corr", y="prompt",
            data=pd_data, orient='h', width=.6,
            boxprops={'color': '#404040',
                      'facecolor': '#d6ecfa'
                      }
        )
    else:
        ax = sns.boxplot(
            x="prompt", y="corr",
            data=pd_data, width=.3,
            palette="Set2"
        )

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    for line in ax.get_lines():
        line.set_color("#404040")

    set_plt()
    if ylim is not None:
        ax.set(ylim=ylim)
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    fig = ax.get_figure()
    fig.savefig('{}/{}.eps'.format(pic_dir, pic_name), format='eps')


def draw_corr_scatter(data, pic_name, pic_dir, prompt="T_{man}"):
    pd_data = pd.DataFrame(data)
    pd_data = pd_data[pd_data["prompts"] == prompt]
    print(pd_data)
    ax = sns.regplot(x="kl", y="precision", data=pd_data)
    print("mean: {}".format(pd_data["kl"].mean()))


def draw_context_box_plot(true_p, false_p, obj_true_p, obj_false_p):
    data = {"prediction": [], "context": [], "precision": []}
    for p in true_p:
        data["precision"].append(p)
        data["prediction"].append("right")
        data["context"].append("mask obj")
    for p in false_p:
        data["precision"].append(p)
        data["prediction"].append("false")
        data["context"].append("mask obj")

    for p in obj_true_p:
        data["precision"].append(p)
        data["prediction"].append("right")
        data["context"].append("obj only")
    for p in obj_false_p:
        data["precision"].append(p)
        data["prediction"].append("false")
        data["context"].append("obj only")

    pd_data = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(
        x="prediction", y="precision", hue="context",
        data=pd_data, palette="Set3", width=.3
    )


def load_wikidata(data_name, relation_id):
    return load_json_dic("data/wikidata/{}/{}".format(data_name, relation_id))


def delete_overlap_by_lower(samples):
    temp_samples = []
    for sample in samples:
        sub, obj = get_pair(sample)
        sub = sub.lower()
        obj = obj.lower()
        if obj in sub:
            continue
        else:
            temp_samples.append(sample)
    return temp_samples


def set_plt():
    config = {
        "font.family": 'serif',
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)


def count_distinct_obj(samples):
    objs = set()
    for sample in samples:
        sub, obj = get_pair(sample)
        if obj not in objs:
            objs.add(obj)
    return len(objs)


def main():
    data = [[0.1, 0.5], [-0.2], [-0.9]]
    row_labels = ['SM', 'OM']
    col_labels = ['PS', 'RS']
    draw_heat_map(data, row_labels, col_labels)


if __name__ == '__main__':
    main()
