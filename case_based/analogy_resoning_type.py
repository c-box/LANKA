from utils.elasticsearch_client import ElasticsearchClient
from utils.utils import build_model, get_table_stat, get_relation_args, \
    load_wikidata, delete_overlap_by_lower, mean_round, model_prefix, set_seed, load_roberta_vocab, \
    store_json_dic, load_json_dic
from prettytable import PrettyTable
from case_based.case_utils import get_vocab_to_type_set, get_relation_token, \
    get_few_shot_samples, sort_samples, \
    get_raw_samples_prediction, few_shot_prediction, analysis_raw_and_few_shot_by_type_inc, \
    analysis_raw_and_few_shot_by_type_rank, analysis_type_prediction, compute_type_precision, \
    compute_conditional_type_prob, analysis_type_rank_change, filter_obj, analysis_type_rank_k
from utils.read_data import LamaDataset
from models import build_model_wrapper
import random
import os


def type_precision(args):
    set_seed(0)

    args = get_relation_args(args)

    table = PrettyTable(
        field_names=["relation_id",
                     "relation_label",
                     "raw_p", "few_p", "p inc",
                     "raw_type_p", "few_type_p", "t inc",
                     "ftr_num",
                     "ftr_false_type",
                     "ftr_overlap",
                     "rtf_num",
                     "rtf_right_type"
                     ]
    )

    table.title = "analogy result model:{} shot:{} choose:{} few-shot prompt: {}".format(
        model_prefix(args.model_name), args.num_of_shot, args.choose_method, args.few_shot_prompt
    )

    es = ElasticsearchClient()
    es.check_connection()

    model_wrapper = build_model_wrapper(args.model_name, device=args.cuda_device)
    if model_prefix(args.model_name) == "bert":
        vocab = model_wrapper.tokenizer.vocab
        vocab2type = get_vocab_to_type_set(es, vocab, model_name=args.model_name)
    elif model_prefix(args.model_name) == "roberta":
        vocab = load_roberta_vocab()
        bert_vocab = get_vocab_to_type_set(es, vocab, model_name="bert")
        vocab2type = get_vocab_to_type_set(es, vocab, args.model_name, bert_vocab)
    else:
        raise RuntimeError("model error")

    lama_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)
    id2relation, relation2samples = lama_data.get_samples()

    relation_tokens = get_relation_token(vocab2type, id2relation, relation2samples, es, args)

    for relation_id in id2relation:
        relation = id2relation[relation_id]
        relation_label = relation["label"]
        relation_template = relation["template"]
        samples = relation2samples[relation_id]

        relation_type_token = relation_tokens[relation_id]
        raw_sample_prediction, raw_precision = \
            get_raw_samples_prediction(args, relation, samples, model_wrapper, topk=args.topk)

        few_shot_samples = []
        if "confidence" in args.choose_method:
            sorted_samples = sort_samples(raw_sample_prediction)
        elif "random" in args.choose_method:
            sorted_samples = filter_obj(raw_sample_prediction)
            sorted_samples = random.sample(sorted_samples, k=min(args.num_of_shot, len(sorted_samples)))
        else:
            raise RuntimeError("method error")
        for sample in samples:
            few_shot_sample = get_few_shot_samples(args, sample, sorted_samples)
            few_shot_samples.append(few_shot_sample)

        few_shot_p, few_shot_results, false_to_right_samples, right_to_false_samples = \
            few_shot_prediction(args, relation, raw_sample_prediction, few_shot_samples,
                                model_wrapper, return_samples=True, return_p=True)

        raw_type_p, raw_false_type_p = compute_type_precision(
            raw_sample_prediction, prediction_type="raw", relation_token=relation_type_token
        )
        few_shot_type_p, few_shot_false_type_p = compute_type_precision(
            few_shot_results, prediction_type="few-shot", relation_token=relation_type_token
        )

        ftr_false_type, ftr_overlap, ftr_type_1, rtf_right_type, \
        ftr_exclude, ftr_type_1_exclude, rtf = \
            compute_conditional_type_prob(
                raw_sample_prediction, few_shot_results,
                false_to_right_samples, right_to_false_samples,
                relation_type_token
            )

        table.add_row([
            relation_id, relation_label,
            raw_precision, few_shot_p, round(few_shot_p - raw_precision, 2),
            raw_type_p, few_shot_type_p, round(few_shot_type_p - raw_type_p, 2),
            len(false_to_right_samples),
            ftr_false_type, ftr_overlap,
            len(right_to_false_samples),
            rtf_right_type
        ])

    table.sortby = "p inc"
    table.reversesort = True
    print(table)
    return table


def type_rank_change(args):
    set_seed(0)
    args = get_relation_args(args)

    table = PrettyTable(
        field_names=["relation_id", "relation_label",
                     "inc", "unchange", "dec", "inc_per", "un_per", "dec_per",
                     "inc_in_few_shot"]
    )
    table.title = "analogy result model:{} shot:{} choose:{} few-shot prompt: {}".format(
        model_prefix(args.model_name), args.num_of_shot, args.choose_method, args.few_shot_prompt
    )

    es = ElasticsearchClient()
    es.check_connection()

    model_wrapper = build_model_wrapper(args.model_name, device=args.cuda_device)
    if model_prefix(args.model_name) == "bert":
        vocab = model_wrapper.tokenizer.vocab
        vocab2type = get_vocab_to_type_set(es, vocab, model_name=args.model_name)
    elif model_prefix(args.model_name) == "roberta":
        vocab = load_roberta_vocab()
        bert_vocab = get_vocab_to_type_set(es, vocab, model_name="bert")
        vocab2type = get_vocab_to_type_set(es, vocab, args.model_name, bert_vocab)
    else:
        raise RuntimeError("model error")

    lama_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)
    id2relation, relation2samples = lama_data.get_samples()

    relation_tokens = get_relation_token(vocab2type, id2relation, relation2samples, es, args)

    inc_total = 0
    unchange_total = 0
    dec_total = 0
    total = 0

    for relation_id in id2relation:
        relation = id2relation[relation_id]
        relation_label = relation["label"]
        relation_template = relation["template"]
        samples = relation2samples[relation_id]

        relation_type_token = relation_tokens[relation_id]

        raw_sample_prediction, raw_precision = \
            get_raw_samples_prediction(args, relation, samples, model_wrapper, topk=args.topk)

        few_shot_samples = []
        if "confidence" in args.choose_method:
            sorted_samples = sort_samples(raw_sample_prediction)
        elif "random" in args.choose_method:
            sorted_samples = filter_obj(raw_sample_prediction)
            sorted_samples = random.sample(sorted_samples, k=min(args.num_of_shot, len(sorted_samples)))
        else:
            raise RuntimeError("method error")
        for sample in samples:
            few_shot_sample = get_few_shot_samples(args, sample, sorted_samples)
            few_shot_samples.append(few_shot_sample)

        few_shot_p, few_shot_results, false_to_right_samples, right_to_false_samples = \
            few_shot_prediction(args, relation, raw_sample_prediction, few_shot_samples,
                                model_wrapper, return_samples=True, return_p=True)

        inc, unchange, dec, total_change, inc_in_few_shot = analysis_type_rank_change(
            raw_sample_prediction, few_shot_results, relation_type_token,
            few_shot_samples, relation_label
        )

        inc_per = mean_round(inc, total_change)
        unchange_per = mean_round(unchange, total_change)
        dec_per = mean_round(dec, total_change)

        inc_total += inc
        unchange_total += unchange
        dec_total += dec

        total += total_change

        table.add_row([relation_id, relation_label, inc, unchange, dec,
                       inc_per, unchange_per, dec_per, inc_in_few_shot])
    print(args)
    table = get_table_stat(table)
    print(table)
    return table
