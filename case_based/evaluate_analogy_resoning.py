from prettytable import PrettyTable
import random
from utils.utils import build_model, get_table_stat, get_relation_args, model_prefix, set_seed
from utils.read_data import LamaDataset
from case_based.case_utils import get_raw_samples_prediction, sort_samples, \
    get_few_shot_samples, few_shot_prediction, filter_obj
from models import build_model_wrapper


def evaluate_analogy_reasoning(args):
    print(args)
    set_seed(0)
    args = get_relation_args(args)

    table = PrettyTable(
        field_names=["relation_id", "relation_label",
                     "old-p@1", "new-p@1", "false_to_right", "right_to_false",
                     "raw_mrr", "few_mrr",
                     "obj_rank_increase", "obj_rank_unchange", "obj_rank_decrease",
                     ]
    )

    table.title = "analogy result model:{} shot:{} choose:{} few-shot prompt: {}".format(
        model_prefix(args.model_name), args.num_of_shot, args.choose_method, args.few_shot_prompt
    )

    model_wrapper = build_model_wrapper(args.model_name, device=args.cuda_device)

    lama_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)

    id2relation, relation2samples = lama_data.get_samples()

    for relation_id in id2relation:
        relation = id2relation[relation_id]
        relation_label = relation['label']

        samples = relation2samples[relation_id]

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

        precision, raw_mrr, few_mrr, \
            obj_rank_to_increase, obj_rank_unchange, obj_rank_to_decrease, \
            false_to_right, right_to_false, few_shot_results = \
            few_shot_prediction(args, relation, raw_sample_prediction, few_shot_samples, model_wrapper)

        table.add_row([
            relation_id, relation_label,
            raw_precision, precision, false_to_right, right_to_false,
            raw_mrr, few_mrr,
            obj_rank_to_increase, obj_rank_unchange, obj_rank_to_decrease
        ])
    table = get_table_stat(table)
    print(args)
    print(table)
    return table
