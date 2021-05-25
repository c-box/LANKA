from utils.utils import get_pair, store_json_dic
import random
from scipy.stats import pearsonr
import scipy
from utils.utils import load_json_dic, model_prefix
import json
import os
import numpy as np
import seaborn as sns
from models.mlm_wrapper import MLMWrapper


def compare_distribution(in_distribution, out_distribution):
    all_tokens = {}
    in_tf = []
    out_tf = []
    for token in in_distribution:
        if token not in all_tokens:
            all_tokens[token] = 0
    for token in out_distribution:
        if token not in all_tokens:
            all_tokens[token] = 0

    for token in all_tokens:
        if token in in_distribution:
            in_tf.append(in_distribution[token])
        else:
            in_tf.append(0)

        if token in out_distribution:
            out_tf.append(out_distribution[token])
        else:
            out_tf.append(0)
    pearsonr_corr = pearsonr(in_tf, out_tf)
    corr = round(pearsonr_corr[0], 4)
    return corr


def topk_dis(dis, topk):
    sorted_dis = sort_dis(dis)
    sorted_dis = sorted_dis[:topk]
    dis = {}
    for d in sorted_dis:
        dis[d[0]] = d[1]
    return dis


def norm_list(lst):
    lst_sum = sum(lst)
    return [(x / lst_sum) for x in lst]


def calculate_kl(in_distribution, out_distribution):
    all_tokens = {}
    in_tf = []
    out_tf = []
    for token in in_distribution:
        if token not in all_tokens:
            all_tokens[token] = 1e-8
    for token in out_distribution:
        if token not in all_tokens:
            all_tokens[token] = 1e-8

    for token in all_tokens:
        if token in in_distribution:
            in_tf.append(in_distribution[token])
        else:
            in_tf.append(1e-8)

        if token in out_distribution:
            out_tf.append(out_distribution[token])
        else:
            out_tf.append(1e-8)
    in_tf = norm_list(in_tf)
    out_tf = norm_list(out_tf)
    KL = scipy.stats.entropy(in_tf, out_tf)
    KL = round(KL, 4)
    return KL


def get_predict_distribution(args, relation, samples, model_wrapper: MLMWrapper):
    distribution = {}

    predict_results, p = model_wrapper.evaluate_samples(
        relation, samples, mask_pos=0, batch_size=args.batch_size,
        topk=args.topk, max_len=args.max_len
    )

    for i in range(len(predict_results)):
        predict_tokens = predict_results[i]['predict_tokens']
        topk_tokens = predict_tokens[: 1]
        for token in topk_tokens:
            if token not in distribution:
                distribution[token] = 0
            distribution[token] += 1
    return distribution


def get_mask_distribution(args, relation, model_wrapper: MLMWrapper, return_topk=False):
    mask_distribution = {}
    input_sentences = []
    relation_template = relation['template']
    if relation_template.find("[X]") < relation_template.find("[Y]"):
        mask_pos = [-1]
    else:
        mask_pos = [0]
    input_sentence = relation_template.replace('[X]', model_wrapper.tokenizer.mask_token)
    input_sentence = input_sentence.replace('[Y]', model_wrapper.tokenizer.mask_token)
    input_sentences.append(input_sentence)
    predict_results = model_wrapper.mlm_predict(
        input_sentences, mask_pos=mask_pos, batch_size=args.batch_size,
        topk=args.mask_topk, max_len=args.max_len
    )
    predict_tokens = predict_results[0]['predict_tokens']
    predict_prob = predict_results[0]['predict_prob']
    topk_tokens = predict_tokens[: args.mask_topk]
    for token, prob in zip(topk_tokens, predict_prob):
        mask_distribution[token] = float(prob)

    if return_topk is True:
        return topk_tokens
    else:
        return mask_distribution


def calculate_prompt_only_dis_kl_div(args, prompt, lama_dis, model_wrapper):
    mask_distribution = {}
    if "[MASK]" not in prompt:
        raise RuntimeError("at least one [MASK] token")

    predict_results = model_wrapper.mlm_predict(
        [prompt], mask_pos=-1, batch_size=args.batch_size,
        topk=args.topk, max_len=args.max_len
    )

    predict_tokens = predict_results[0]['predict_tokens']
    predict_prob = predict_results[0]['predict_prob']
    topk_tokens = predict_tokens[: args.mask_topk]

    for token, prob in zip(topk_tokens, predict_prob):
        mask_distribution[token] = float(prob)

    mask_dis = topk_dis(mask_distribution, 1000)
    return calculate_kl(mask_dis, lama_dis)


def get_obj_distribution(samples):
    distribution = {}
    for sample in samples:
        sub, obj = get_pair(sample)
        if obj not in distribution:
            distribution[obj] = 0
        distribution[obj] += 1
    return distribution


def delete_overlap(samples):
    temp_samples = []
    for sample in samples:
        sub, obj = get_pair(sample)
        if obj in sub:
            continue
        else:
            temp_samples.append(sample)
    return temp_samples


def devide_by_vocab(samples, vocab):
    in_samples = []
    not_in_samples = []
    for sample in samples:
        if 'sample' in sample:
            sample = sample['sample']
        sub = sample['sub_label']
        obj = sample['obj_label']
        if obj in vocab:
            in_samples.append(sample)
        else:
            not_in_samples.append(sample)
    return in_samples, not_in_samples


def store_samples(relation_id, out_dir, samples):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open("{}/{}".format(out_dir, relation_id), "w") as f:
        json.dump(samples, f)


def average_sampling(samples, method, threshold, target):
    sampled_cases = []
    distribution = get_obj_distribution(samples)

    fre = []
    for obj in distribution:
        fre.append(distribution[obj])
    diff_obj = len(distribution)
    fre = sorted(fre)

    down_quantile_fre = int(np.quantile(fre, threshold))

    down_target_fre = -1
    for i in range(diff_obj, 0, -1):
        idx = diff_obj - i
        if fre[idx] * i >= target:
            down_target_fre = fre[idx]
            break
    if down_target_fre == -1:
        down_target_fre = down_quantile_fre

    if method == "threshold_sample":
        lower_bound = down_quantile_fre
    else:
        lower_bound = down_target_fre

    obj2samples = {}
    for sample in samples:
        if 'sample' in sample:
            sample = sample['sample']
        obj = sample['obj_label']
        if obj not in obj2samples:
            obj2samples[obj] = []
        obj2samples[obj].append(sample)
    for obj in obj2samples:
        if len(obj2samples[obj]) >= lower_bound:
            sampled_cases.extend(random.sample(obj2samples[obj], lower_bound))
    return sampled_cases


def load_wiki_uni(relation_id, model="bert"):
    return load_json_dic("data/{}_data/wiki_uni/{}".format(model, relation_id))


def store_dis(dis_dir, prompt_type,  dis_type, relation_id, data):
    if not os.path.isdir("{}/{}/{}".format(dis_dir, prompt_type, dis_type)):
        os.makedirs("{}/{}/{}".format(dis_dir, prompt_type, dis_type))

    store_json_dic("{}/{}/{}/{}".format(dis_dir, prompt_type, dis_type, relation_id), data)


def load_dis(dis_dir, prompt_type, dis_type, relation_id):
    return load_json_dic("{}/{}/{}/{}".format(dis_dir, prompt_type, dis_type, relation_id))


def add_corr(corr_dic, relation_type, corr):
    if "original" in relation_type:
        corr_dic[r"$T_{man}$"].append(corr)
    elif "mine" in relation_type:
        corr_dic[r"$T_{mine}$"].append(corr)
    elif "auto" in relation_type:
        corr_dic[r"$T_{auto}$"].append(corr)


def sort_dis(distribution):
    return sorted(distribution.items(), key=lambda x: x[1], reverse=True)


def sum_dis(dis):
    ans = 0
    for token in dis:
        ans += dis[token]
    return ans


def check_args(args):
    if model_prefix(args.model_name) == "roberta":
        assert "roberta" in args.relation_type, "relation type don't match the model"
    elif model_prefix(args.model_name) == "bert":
        assert "lama" in args.relation_type, "relation type don't match the model"
    else:
        raise RuntimeError("wrong model")
