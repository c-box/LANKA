from utils.utils import build_model, get_relation_args, get_pair, \
    set_seed, get_table_stat, mean_round, model_prefix
from utils.read_data import LamaDataset
from prettytable import PrettyTable
from utils.mlm_predict import mlm_predict
import re
from context_based.context_utils import get_new_context, get_result_with_context
from utils.utils import filter_samples_by_vocab, load_roberta_vocab
from models import build_model_wrapper


def explicit_leak(args):
    set_seed(0)
    table = PrettyTable(
        field_names=["relation_id", "relation_label",
                     "none",
                     "drqa", "mask_drqa"]
    )
    table.title = "overall performance - {}".format(args.model_name)
    # tokenizer, model = build_model(args.model_name)
    model_wrapper = build_model_wrapper(args.model_name, args.cuda_device)

    args.relation_type = "lama_original"
    args = get_relation_args(args)
    lama_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)
    id2relation, relation2samples = lama_data.get_samples()

    args.relation_type = "lama_orginal_with_lama_drqa"
    args = get_relation_args(args)
    draq_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)
    _, id2drqa_samples = draq_data.get_samples()

    case_tables = []
    table_names = ["drqa"]

    for i in range(len(table_names)):
        case_table = PrettyTable(
            field_names=[
                "id", "label",
                "present",
                "present_without", "present_with", "present_mask_with",
                "absent",
                "absent_without", "absent_with", "absent_mask_with"
            ]
        )
        case_table.title = table_names[i]
        case_tables.append(case_table)

    for relation_id in id2relation:
        relation = id2relation[relation_id]
        relation_label = relation["label"]
        samples = relation2samples[relation_id]
        drqa_samples = id2drqa_samples[relation_id]
        samples = align_samples(samples, drqa_samples)

        if model_prefix(args.model_name) == "roberta":
            samples = filter_samples_by_vocab(samples, load_roberta_vocab())[0]

        # results without context
        st_p, st_ans = get_result_with_context(
            args, relation, samples, model_wrapper,
            return_samples=False, no_context=True,
            return_ans=True
        )
        new_row = [relation_id, relation_label, st_p]

        context_methods = [
            "lama_drqa"
        ]

        table_idx = 0
        for context_method in context_methods:
            # results with context
            args.context_method = context_method
            args.mask_obj = False
            cst_p, cst_ans, obj_in_contexts = get_result_with_context(
                args, relation, samples, model_wrapper,
                return_ans=True, return_obj_in_context=True
            )
            new_row.append(cst_p)

            # results with masked context
            args.mask_obj = True
            mask_p, mask_ans = get_result_with_context(
                args, relation, samples, model_wrapper,
                return_ans=True
            )
            new_row.append(mask_p)

            present = 0
            absent = 0
            pre_without = 0
            pre_with = 0
            mask_pre_with = 0
            ab_without = 0
            ab_with = 0
            mask_ab_with = 0

            for i in range(len(st_ans)):
                if obj_in_contexts[i]:
                    present += 1
                    if st_ans[i]:
                        pre_without += 1
                    if cst_ans[i]:
                        pre_with += 1
                    if mask_ans[i]:
                        mask_pre_with += 1
                else:
                    absent += 1
                    if st_ans[i]:
                        ab_without += 1
                    if cst_ans[i]:
                        ab_with += 1
                    if mask_ans[i]:
                        mask_ab_with += 1
            case_tables[table_idx].add_row([
                relation_id, relation_label,
                present,
                mean_round(pre_without, present),
                mean_round(pre_with, present), mean_round(mask_pre_with, present),
                absent,
                mean_round(ab_without, absent),
                mean_round(ab_with, absent), mean_round(mask_ab_with, absent)
            ])
            table_idx += 1
        table.add_row(new_row)
        print(table)

    table = get_table_stat(table)
    print(table)

    for case_table in case_tables:
        case_table = get_table_stat(case_table)
        print(case_table)


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
