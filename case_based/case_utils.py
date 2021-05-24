from utils.mlm_predict import mlm_prdict_results, mlm_predict
from utils.utils import get_pair, mean_round, get_table_stat, delete_overlap_by_lower, model_prefix, load_roberta_vocab
from prettytable import PrettyTable
import os
import json
from tqdm import tqdm
from utils.search import search_for_class, bfs_class_set, search_sample_type
import numpy as np
from models.mlm_wrapper import MLMWrapper


def get_raw_samples_prediction(args, relation, samples, model_wrapper: MLMWrapper, topk=10):
    sample_prediction = []

    results, p_1 = model_wrapper.evaluate_samples(
        relation=relation, samples=samples, pass_obj=True,
        mask_pos=0, batch_size=args.batch_size, topk=topk, max_len=args.max_len
    )

    for i in range(len(samples)):
        sample = samples[i]
        result = results[i]
        topk = []
        for k in range(len(result['predict_tokens'])):
            topk.append(
                {
                    "i": k,
                    "token_word_form": result['predict_tokens'][k],
                    "prob": round(float(result['predict_prob'][k]), 4)
                }
            )
        predict_sample = {
            "sample": sample, "topk": topk,
            "predict_ans": result['predict_ans'],
            'obj_prob': result['obj_prob'],
            'obj_rank': result['obj_rank'],
            'mrr': result['mrr'],
        }
        sample_prediction.append(predict_sample)
    return sample_prediction, p_1


def get_few_shot_samples(args, sample, sorted_prediction):
    few_shot_sample = []
    if 'without' in args.choose_method:
        sub, obj = get_pair(sample)
        few_shot_objs = []
        for raw_sample in sorted_prediction:
            few_shot_sub, few_shot_obj = get_pair(raw_sample)
            if few_shot_obj != obj and few_shot_obj not in few_shot_objs:
                few_shot_sample.append(raw_sample)
                few_shot_objs.append(few_shot_obj)
            if len(few_shot_sample) == args.num_of_shot:
                break
        return few_shot_sample
    elif "with" in args.choose_method:
        sub, obj = get_pair(sample)
        few_shot_objs = [obj]
        few_shot_sample.append(search_sample_by_obj(sorted_prediction, sample))
        for raw_sample in sorted_prediction:
            if len(few_shot_sample) == args.num_of_shot:
                break
            few_shot_sub, few_shot_obj = get_pair(raw_sample)
            if few_shot_obj not in few_shot_objs:
                few_shot_sample.append(raw_sample)
                few_shot_objs.append(few_shot_obj)
        return few_shot_sample
    else:
        few_shot_objs = []
        for raw_sample in sorted_prediction:
            few_shot_sub, few_shot_obj = get_pair(raw_sample)
            if few_shot_obj not in few_shot_objs:
                few_shot_sample.append(raw_sample)
                few_shot_objs.append(few_shot_obj)
            if len(few_shot_sample) == args.num_of_shot:
                break
        return few_shot_sample


def search_sample_by_obj(samples, sample):
    sub, obj = get_pair(sample)
    for search_sample in samples:
        search_sub, search_obj = get_pair(search_sample)
        if search_sub != sub and search_obj == obj:
            return search_sample
    return sample


def sort_samples(samples, sort_by='obj_prob'):
    if sort_by == "obj_prob":
        return sorted(samples, key=lambda x: x[sort_by], reverse=True)
    else:
        raise RuntimeError("no key")


def samples_to_context(args, samples, template, mask_token='[MASK]'):
    context = ""
    for sample in samples:
        sub, obj = get_pair(sample)

        if args.few_shot_prompt == 'and':
            sentence = "{} and {} .".format(sub, obj)
            context = context + " " + sentence
        elif args.few_shot_prompt == 'original':
            sentence = template.replace('[X]', sub)
            sentence = sentence.replace('[Y]', obj)
            context = context + " " + sentence
        elif args.few_shot_prompt == 'obj':
            context = context + " " + obj
        elif args.few_shot_prompt == 'mask_sub':
            sentence = template.replace('[X]', mask_token)
            sentence = sentence.replace('[Y]', obj)
            context = context + " " + sentence
        elif args.few_shot_prompt == 'no_sub':
            sentence = template.replace('[X]', "")
            sentence = sentence.replace('[Y]', obj)
            context = context + " " + sentence
        else:
            raise RuntimeError("no such prompt")
    return context.strip(' ')


def get_resoning_results(args, relation,
                         samples_to_predict, few_shot_samples,
                         model_wrapper: MLMWrapper):
    relation_template = relation['template']
    input_sentences = []
    obj_tokens = []
    mask_token = model_wrapper.tokenizer.mask_token
    sep_token = model_wrapper.tokenizer.sep_token
    for i in range(len(samples_to_predict)):
        sample = samples_to_predict[i]
        context = samples_to_context(args, few_shot_samples[i], relation_template,
                                     mask_token=mask_token)
        sub, obj = get_pair(sample)
        obj_tokens.append(obj)
        if args.sample_prompt == 'and':
            input_sentence = "{} and {} .".format(sub, mask_token)
        elif args.sample_prompt == 'original':
            input_sentence = relation_template.replace('[X]', sub)
            input_sentence = input_sentence.replace('[Y]', mask_token)
        else:
            raise RuntimeError("no such prompt")

        if model_prefix(args.model_name) == "bert":
            if args.token_type:
                input_sentence = [context, input_sentence]
            else:
                input_sentence = context + ' {} '.format(sep_token) + input_sentence
        elif model_prefix(args.model_name) == "roberta":
            input_sentence = context + ' {} '.format(sep_token) + input_sentence
        else:
            raise RuntimeError("model error")

        input_sentences.append(input_sentence)

    predict_results = model_wrapper.mlm_predict(
        input_sentences, mask_pos=-1, batch_size=args.batch_size,
        obj_tokens=obj_tokens, topk=args.topk, max_len=args.max_len
    )
    return predict_results


def few_shot_prediction(args, relation,
                        samples_to_predict, few_shot_samples,
                        model_wrapper, return_samples=False, return_p=False):
    predict_results = get_resoning_results(args, relation,
                                           samples_to_predict, few_shot_samples,
                                           model_wrapper)
    precision = 0
    obj_rank_to_increase = 0
    obj_rank_to_decrease = 0
    obj_rank_unchange = 0
    raw_mrr = 0
    few_mrr = 0
    false_to_right = 0
    right_to_false = 0

    false_to_right_samples = []
    right_to_false_samples = []

    for i in range(len(predict_results)):
        result = predict_results[i]
        sample = samples_to_predict[i]
        if result['predict_ans'] is True:
            precision += 1
            if sample['predict_ans'] is False:
                false_to_right += 1
                false_to_right_samples.append(sample)
        else:
            if sample['predict_ans'] is True:
                right_to_false += 1
                right_to_false_samples.append(sample)
        if result['obj_rank'] < sample['obj_rank']:
            obj_rank_to_increase += 1
        if result["obj_rank"] > sample["obj_rank"]:
            obj_rank_to_decrease += 1
        if result["obj_rank"] == sample["obj_rank"]:
            obj_rank_unchange += 1
        raw_mrr += 1.0 / sample["obj_rank"]
        few_mrr += 1.0 / result["obj_rank"]

    num_len = len(predict_results)
    percent_precision = mean_round(precision, num_len)
    raw_mrr = round(raw_mrr / num_len, 4)
    few_mrr = round(few_mrr / num_len, 4)
    obj_rank_to_increase = mean_round(obj_rank_to_increase, num_len)
    obj_rank_to_decrease = mean_round(obj_rank_to_decrease, num_len)
    obj_rank_unchange = mean_round(obj_rank_unchange, num_len)
    false_to_right = mean_round(false_to_right, num_len)
    right_to_false = mean_round(right_to_false, num_len)
    if return_p is True:
        return percent_precision, predict_results, false_to_right_samples, right_to_false_samples
    else:
        if return_samples is True:
            return predict_results, false_to_right_samples, right_to_false_samples
        else:
            return percent_precision, raw_mrr, few_mrr, \
                   obj_rank_to_increase, obj_rank_unchange, obj_rank_to_decrease,\
                   false_to_right, right_to_false, predict_results


def samples_to_objs(samples):
    objs = []
    for sample in samples:
        sub, obj = get_pair(sample)
        # print(sample)
        objs.append(obj)
    return objs


def analysis_raw_and_few_shot(raw_predictions, few_shot_predictions, few_shot_samples, k=1):
    prediction_in_few_shot = 0
    right_in_few_shot = 0
    false_to_right_in_few_shot = 0

    for i in range(len(raw_predictions)):
        raw_prediction = raw_predictions[i]
        few_shot_prediction = few_shot_predictions[i]
        few_shot_sample = few_shot_samples[i]
        few_shot_objs = samples_to_objs(few_shot_sample)

        few_shot_tokens = few_shot_prediction["predict_tokens"][: k]
        for token in few_shot_tokens:
            if token in few_shot_objs:
                prediction_in_few_shot += 1
                if few_shot_prediction['predict_ans'] is True:
                    right_in_few_shot += 1
                    if raw_prediction['predict_ans'] is False:
                        false_to_right_in_few_shot += 1
                break
    num_len = len(raw_predictions)
    prediction_in_few_shot = mean_round(prediction_in_few_shot, num_len)
    right_in_few_shot = mean_round(right_in_few_shot, num_len)
    false_to_right_in_few_shot = mean_round(false_to_right_in_few_shot, num_len)
    return prediction_in_few_shot, right_in_few_shot, false_to_right_in_few_shot


def compute_type_probs(tokens, probs, relation_token):
    type_num = 0
    type_probs = 0
    for token, prob in zip(tokens, probs):
        if token in relation_token:
            type_num += 1
            type_probs += prob
    return type_num, type_probs


def analysis_raw_and_few_shot_by_type_inc(raw_predictions, few_shot_predictions,
                                          relation_token, topk=10):
    rtm = []
    rtp = []
    ftm = []
    ftp = []
    for i in range(len(raw_predictions)):
        raw_prediction = raw_predictions[i]
        few_shot_prediction = few_shot_predictions[i]

        raw_tokens = []
        raw_probs = []
        for k in range(topk):
            raw_tokens.append(raw_prediction["topk"][k]["token_word_form"])
            raw_probs.append(raw_prediction["topk"][k]["prob"])

        few_shot_tokens = few_shot_prediction["predict_tokens"][: topk]
        few_shot_probs = few_shot_prediction["predict_prob"][: topk]

        raw_type_num, raw_type_probs = compute_type_probs(raw_tokens, raw_probs, relation_token)
        few_shot_type_num, few_shot_type_probs = compute_type_probs(few_shot_tokens, few_shot_probs, relation_token)
        rtm.append(raw_type_num)
        rtp.append(raw_type_probs)
        ftm.append(few_shot_type_num)
        ftp.append(few_shot_type_probs)

    return round(float(np.mean(rtm)), 2), round(float(np.mean(rtp)), 2), \
           round(float(np.mean(ftm)), 2), round(float(np.mean(ftp)), 2)


def in_samples(sub, obj, samples):
    for sample in samples:
        s, o = get_pair(sample)
        if s == sub and o == obj:
            return True
    return False


def analysis_raw_and_few_shot_by_type_rank(raw_predictions, few_shot_predictions,
                                           false_to_right_samples, right_to_false_samples,
                                           relation_token, vocab2type,
                                           relation_id, relation_label):

    false_to_right_table = PrettyTable(
        field_names=["sub", "obj", "raw_rank", "raw_type_rank", "type front"]
    )
    false_to_right_table.title = "{} {} false to right".format(relation_id, relation_label)
    right_to_false_table = PrettyTable(
        field_names=["sub", "obj", "cur prediction", "in_type"]
    )
    right_to_false_table.title = "{} {} right to false".format(relation_id, relation_label)

    for i in range(len(raw_predictions)):
        raw_prediction = raw_predictions[i]
        few_shot_prediction = few_shot_predictions[i]
        sub, obj = get_pair(raw_prediction)

        if in_samples(sub, obj, false_to_right_samples):
            topk = raw_prediction["topk"]
            raw_rank = raw_prediction["obj_rank"]
            raw_type_rank = 0
            front_tokens = []
            if raw_rank >= len(topk):
                raw_type_rank = 10000
            else:
                for k in range(raw_rank):
                    token = topk[k]["token_word_form"]
                    if token in relation_token:
                        raw_type_rank += 1
                        front_tokens.append(token)
            front_tokens = front_tokens[:10]
            false_to_right_table.add_row([sub, obj, raw_rank, raw_type_rank, " ".join(front_tokens)])

        if in_samples(sub, obj, right_to_false_samples):
            token = few_shot_prediction["predict_tokens"][0]
            if token in relation_token:
                in_type = True
            else:
                in_type = False

            type_set = vocab2type[token]
            types = []
            for token_type in type_set:
                type_label = token_type["label"]
                types.append(type_label)
            right_to_false_table.add_row([sub, obj, token, in_type])

    false_to_right_table = get_table_stat(false_to_right_table)
    right_to_false_table = get_table_stat(right_to_false_table)
    print(false_to_right_table)
    print(right_to_false_table)


def get_vocab_to_type_set(es, vocab, model_name, bert_vocab=None):
    if model_prefix(model_name) == "bert":
        vocab2type_file = 'data/type_file/brand_new_bert_vocab_to_types'
    elif model_prefix(model_name) == "roberta":
        assert bert_vocab is not None
        vocab2type_file = 'data/type_file/brand_new_roberta_vocab_to_types'
    else:
        raise RuntimeError("model error")

    if os.path.isfile(vocab2type_file):
        with open(vocab2type_file, 'r') as f:
            vocab2type = json.load(f)
            return vocab2type
    vocab2type = {}
    for word in tqdm(vocab):
        if bert_vocab is not None and word in bert_vocab:
            vocab2type[word] = bert_vocab[word]
        else:
            p31_class_set = search_for_class(es, entity_name=word, index="wikidata_tuples_p31")
            p279_class_set = search_for_class(es, entity_name=word, index="wikidata_tuples_p279")
            class_set = p31_class_set + p279_class_set
            token_types = bfs_class_set(class_set, es)
            vocab2type[word] = token_types
    with open(vocab2type_file, 'w') as f:
        json.dump(vocab2type, f)
    return vocab2type


stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
]


def get_relation_token(vocab2type, id2relation, relation2samples, es, args):

    if model_prefix(args.model_name) == "bert":
        relation_token_file = 'data/type_file/new_bert_relation_token.json'
    elif model_prefix(args.model_name) == "roberta":
        relation_token_file = 'data/type_file/new_roberta_relation_token.json'
    else:
        raise RuntimeError("model error")

    if os.path.isfile(relation_token_file):
        with open(relation_token_file, 'r') as f:
            relation_token = json.load(f)
            for relation_id in relation_token:
                for word in stop_words:
                    if word in relation_token[relation_id]:
                        relation_token[relation_id].remove(word)
            return relation_token

    relation_token = {}
    for relation_id in id2relation:
        relation_token[relation_id] = []
        samples = relation2samples[relation_id]
        obj_types = search_sample_type(es, samples)
        print(relation_id)
        print(obj_types)
        for vocab in tqdm(vocab2type):
            if vocab in stop_words:
                continue
            token_types = vocab2type[vocab]
            for class_type in token_types:
                if class_type['id'] in obj_types:
                    relation_token[relation_id].append(vocab)
                    break

    with open(relation_token_file, 'w') as f:
        json.dump(relation_token, f)

    return relation_token


def analysis_type_prediction(predictions, prediction_type,
                             relation_token):
    table = PrettyTable(
        field_names=["prediction", "true", "false", "overall"]
    )

    precisions = [0 for i in range(4)]

    for i in range(len(predictions)):
        prediction = predictions[i]

        if prediction_type == "raw":
            token = prediction["topk"][0]["token_word_form"]
        else:
            token = prediction["predict_tokens"][0]

        if prediction["predict_ans"] is True:
            precisions[0] += 1
        else:
            if token in relation_token:
                precisions[1] += 1
            else:
                precisions[3] += 1

    for j in range(len(precisions)):
        precisions[j] = mean_round(precisions[j], len(predictions))

    table.add_row(["type_right", precisions[0], precisions[1],
                   round(precisions[0]+precisions[1], 2)])
    table.add_row(["type_false", precisions[2], precisions[3],
                   round(precisions[2]+precisions[3], 2)])

    table.add_row(["overall",
                   round(precisions[0]+precisions[2], 2),
                   round(precisions[1]+precisions[3], 2), 1])

    print(table.get_string(title=prediction_type))


def compute_type_precision(predictions, prediction_type, relation_token):
    type_p = 0
    false_type_p = 0

    for i in range(len(predictions)):
        prediction = predictions[i]

        if prediction_type == "raw":
            token = prediction["topk"][0]["token_word_form"]
        else:
            token = prediction["predict_tokens"][0]

        if token in relation_token:
            type_p += 1
            if prediction["predict_ans"] is True:
                false_type_p += 1
    if type_p != 0:
        false_type_p = round(false_type_p * 100 / type_p, 2)
    else:
        false_type_p = 0
    type_p = mean_round(type_p, len(predictions))

    return type_p, false_type_p


def compute_conditional_type_prob(raw_predictions, few_shot_predictions,
                                  false_to_right_samples, right_to_false_samples,
                                  relation_token):
    ftr_false_type = 0
    ftr_overlap = 0
    ftr_type_1 = 0
    rtf_right_type = 0

    ftr_exclude = len(false_to_right_samples)
    ftr_type_1_exclude = len(false_to_right_samples)
    rtf = len(right_to_false_samples)

    for i in range(len(raw_predictions)):
        raw_prediction = raw_predictions[i]
        few_shot_prediction = few_shot_predictions[i]
        sub, obj = get_pair(raw_prediction)

        if in_samples(sub, obj, false_to_right_samples):
            top_token = raw_prediction["topk"][0]["token_word_form"]
            if top_token not in relation_token:
                ftr_false_type += 1
            raw_rank = raw_prediction["obj_rank"]
            if raw_rank > 10000:
                continue
            raw_type_rank = 0
            for k in range(raw_rank - 1):
                token = raw_prediction["topk"][k]["token_word_form"]
                if token in relation_token:
                    raw_type_rank = 1
                    break
            if raw_type_rank == 0:
                ftr_type_1 += 1

            if judge_overlap(sub, obj):
                ftr_overlap += 1

            if top_token not in relation_token or judge_overlap(sub, obj):
                ftr_exclude -= 1

            if raw_type_rank == 0 or judge_overlap(sub, obj):
                ftr_type_1_exclude -= 1

        if in_samples(sub, obj, right_to_false_samples):
            token = few_shot_prediction["predict_tokens"][0]
            if token in relation_token:
                rtf_right_type += 1

    if len(false_to_right_samples) == 0:
        ftr_false_type = 100.01
        ftr_type_1 = 100.01
        ftr_overlap = 100.01
    else:
        ftr_false_type = round(ftr_false_type * 100 / len(false_to_right_samples), 2)
        ftr_type_1 = round(ftr_type_1 * 100 / len(false_to_right_samples), 2)
        ftr_overlap = round(ftr_overlap * 100 / len(false_to_right_samples), 2)

    if len(right_to_false_samples) == 0:
        rtf_right_type = 100
    else:
        rtf_right_type = round(rtf_right_type * 100 / len(right_to_false_samples), 2)

    ftr_exclude = mean_round(ftr_exclude, len(raw_predictions))
    ftr_type_1_exclude = mean_round(ftr_type_1_exclude, len(raw_predictions))
    rtf = mean_round(rtf, len(raw_predictions))

    return ftr_false_type, ftr_overlap, \
           ftr_type_1, rtf_right_type, \
           ftr_exclude, ftr_type_1_exclude, rtf


def judge_overlap(sub, obj):
    sub = sub.lower()
    obj = obj.lower()
    if obj in sub:
        return True
    else:
        return False


def get_type_rank(obj, topk_token, relation_token):
    rank = 1
    for token in topk_token:
        if token == obj:
            return rank
        if token in relation_token:
            rank += 1
    return rank


def judge_lower_overlap(sub, obj):
    sub = sub.lower()
    obj = obj.lower()
    if obj in sub:
        return True
    else:
        return False


def analysis_type_rank_change(raw_predictions, few_shot_predictions, relation_token,
                              few_shot_samples, relation):
    inc = 0
    unchange = 0
    dec = 0
    total = 0

    inc_in_few_shot = 0

    inc_samples = []
    dec_samples = []
    unchange_samples = []

    for i in range(len(raw_predictions)):
        raw_prediction = raw_predictions[i]
        few_shot_prediction = few_shot_predictions[i]
        sub, obj = get_pair(raw_prediction)

        raw_topk_token = []
        for topk in raw_prediction["topk"]:
            raw_topk_token.append(topk["token_word_form"])

        few_shot_topk_token = few_shot_prediction["predict_tokens"]

        raw_type_rank = get_type_rank(obj, raw_topk_token, relation_token)
        few_shot_type_rank = get_type_rank(obj, few_shot_topk_token, relation_token)

        few_shot_objs = samples_to_objs(few_shot_samples[i])

        if raw_type_rank > few_shot_type_rank:
            inc += 1
            inc_samples.append("{} / {} / {} / {}".format(
                sub, obj, raw_type_rank - few_shot_type_rank, raw_topk_token[0]
            ))
            if obj in few_shot_objs:
                inc_in_few_shot += 1
        elif raw_type_rank == few_shot_type_rank:
            unchange += 1
            unchange_samples.append("{} / {} / {} / {}".format(
                sub, obj, raw_type_rank - few_shot_type_rank, raw_topk_token[0]
            ))
        else:
            dec += 1
            dec_samples.append("{} / {} / {}/ {}".format(
                sub, obj, raw_type_rank - few_shot_type_rank, raw_topk_token[0]
            ))
        total += 1

    inc_in_few_shot = mean_round(inc_in_few_shot, inc)

    ana_table = PrettyTable(
        field_names=["inc", "unchange", "dec"]
    )

    ana_table.title = "{} - {} - {} - {} - {}".format(
        relation, inc, unchange, dec, inc_in_few_shot
    )

    for i in range(20):
        ana_table.add_row([
            get_list_str(i, inc_samples),
            get_list_str(i, unchange_samples),
            get_list_str(i, dec_samples)
        ])

    return inc, unchange, dec, total, inc_in_few_shot


def get_list_str(i, x):
    if i >= len(x):
        return "-"
    else:
        return x[i]


def filter_obj(samples):
    filter_samples = []
    objs = []
    for sample in samples:
        sub, obj = get_pair(sample)
        if obj not in objs:
            objs.append(obj)
            filter_samples.append(sample)
    return filter_samples


def analysis_type_rank_k(raw_predictions, few_shot_predictions, relation_token,
                         few_shot_samples, relation):
    topk_num = [1, 3, 5, 10, 20]
    raw_topk_num = [0 for i in range(5)]
    now_topk_num = [0 for i in range(5)]
    for i in range(len(raw_predictions)):
        raw_prediction = raw_predictions[i]
        few_shot_prediction = few_shot_predictions[i]
        sub, obj = get_pair(raw_prediction)

        if judge_overlap(sub, obj):
            continue
        raw_topk_token = []
        for topk in raw_prediction["topk"]:
            raw_topk_token.append(topk["token_word_form"])

        few_shot_topk_token = few_shot_prediction["predict_tokens"]

        raw_type_rank = get_type_rank(obj, raw_topk_token, relation_token)
        few_shot_type_rank = get_type_rank(obj, few_shot_topk_token, relation_token)

        for j in range(len(topk_num)):
            if raw_type_rank <= topk_num[j]:
                raw_topk_num[j] += 1
            if few_shot_type_rank <= topk_num[j]:
                now_topk_num[j] += 1

    for j in range(len(topk_num)):
        raw_topk_num[j] = mean_round(raw_topk_num[j], len(raw_predictions))
        now_topk_num[j] = mean_round(now_topk_num[j], len(raw_predictions))
    return raw_topk_num, now_topk_num


