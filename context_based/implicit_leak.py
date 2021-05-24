from utils.utils import build_model, get_relation_args, get_pair, \
    set_seed, get_table_stat, draw_context_box_plot, mean_round, model_prefix
from utils.read_data import LamaDataset
from prettytable import PrettyTable
from utils.mlm_predict import mlm_predict_with_multi_mask
import re
from context_based.context_utils import get_new_context, get_result_with_context
from utils.utils import filter_samples_by_vocab, load_roberta_vocab
from models import build_model_wrapper


def predict_context_with_multi_mask(args, relation, relation_samples, tokenizer, model):
    relation_template = relation['template']
    input_sentences = []
    gold_obj = []
    assert args.mask_obj is True
    for i in range(len(relation_samples)):
        relation_sample = relation_samples[i]
        sub, obj = get_pair(relation_sample)
        gold_obj.append(obj)
        context = get_new_context(
            args, relation_sample,
            relation_template, tokenizer=tokenizer,
            mask_token=tokenizer.mask_token
        )
        input_sentences.append(context)
    predict_results = mlm_predict_with_multi_mask(
        tokenizer, model, input_sentences,
        batch_size=args.batch_size, args=args
    )
    right_samples = []
    false_samples = []
    for i in range(len(relation_samples)):
        predict_result = predict_results[i]
        sample = relation_samples[i]
        sub, obj = get_pair(sample)
        if obj in predict_result:
            right_samples.append(sample)
        else:
            false_samples.append(sample)
    return right_samples, false_samples


def implicit_leak(args):
    set_seed(0)
    args = get_relation_args(args)
    model_wrapper = build_model_wrapper(args.model_name, args.cuda_device)

    lama_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)
    id2relation, relation2samples = lama_data.get_samples()

    table = PrettyTable(
        field_names=["id", "label",
                     "num_true",
                     "c_true_cst",
                     "c_true_st",
                     "true_inc",
                     "num_false",
                     "c_false_cst",
                     "c_false_st",
                     "false_inc"
                     ]
    )
    table.title = "corr_context_object {}".format(args.context_method)

    for relation_id in id2relation:
        relation = id2relation[relation_id]
        relation_label = relation["label"]
        samples = relation2samples[relation_id]

        if args.context_method == "raw_oracle":
            samples = add_oracle_context(samples)
        samples = filter_context_with_obj(args, samples)

        if model_prefix(args.model_name) == "roberta":
            samples = filter_samples_by_vocab(samples, load_roberta_vocab())[0]

        ct_right, ct_false = \
            predict_context_with_multi_mask(
                args, relation, samples, model_wrapper.tokenizer, model_wrapper.model
            )

        ct_true_st = get_result_with_context(
            args, relation, ct_right, model_wrapper, return_samples=False, no_context=True
        )
        ct_false_st = get_result_with_context(
            args, relation, ct_false, model_wrapper, return_samples=False, no_context=True
        )

        ct_true_cst = get_result_with_context(
            args, relation, ct_right, model_wrapper, return_samples=False
        )
        ct_false_cst = get_result_with_context(
            args, relation, ct_false, model_wrapper, return_samples=False
        )

        table.add_row([
            relation_id, relation_label,
            len(ct_right),
            ct_true_cst, ct_true_st,
            round(ct_true_cst - ct_true_st, 2),
            len(ct_false),
            ct_false_cst, ct_false_st,
            round(ct_false_cst - ct_false_st, 2)
        ])
        print(table)
    table = get_table_stat(table)
    print(table)


def filter_context_with_obj(args, samples):
    filter_samples = []
    mask_obj = args.mask_obj
    args.mask_obj = False
    for sample in samples:
        context = get_new_context(args, sample)
        sub, obj = get_pair(sample)
        if obj in context:
            filter_samples.append(sample)
    args.mask_obj = mask_obj
    return filter_samples


def search_for_samples(samples, sub, obj):
    for sample in samples:
        s, o = get_pair(sample)
        if s == sub and o == obj:
            return True
    return False


def filter_context_with_one_obj(args, samples):
    filter_samples = []
    mask_obj = args.mask_obj
    args.mask_obj = False
    for sample in samples:
        context = get_new_context(args, sample)
        sub, obj = get_pair(sample)
        cnt = 0
        context = context.split()
        for i in range(len(context)):
            if obj in context[i]:
                cnt += 1
        if cnt == 1:
            filter_samples.append(sample)
    args.mask_obj = mask_obj
    return filter_samples


def align_samples(samples, drqa_samples):
    for i in range(len(samples)):
        sample = samples[i]
        sub, obj = get_pair(sample)
        for drqa_sample in drqa_samples:
            s, o = get_pair(drqa_sample)
            if s == sub and o == obj:
                samples[i]["context"] = drqa_sample["context"]
    return samples


def add_oracle_context(samples):
    for i in range(len(samples)):
        sample = samples[i]
        sub, obj = get_pair(sample)
        evidences = sample["sample"]["evidences"]
        evidence = evidences[0]
        sentence = re.sub(re.escape("[MASK]"),
                          evidence["obj_surface"],
                          evidence["masked_sentence"])
        samples[i]["oracle_context"] = [sentence]
    return samples

