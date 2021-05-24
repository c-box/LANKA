from utils.utils import get_relation_args
from utils.read_data import LamaDataset
from prettytable import PrettyTable
from utils.utils import set_seed, get_table_stat
from utils.utils import draw_box_plot, model_prefix, mean_round, set_plt
import matplotlib.pyplot as plt
from prompt_based.prompt_utils import compare_distribution, \
    delete_overlap, get_predict_distribution, \
    get_mask_distribution, get_obj_distribution, \
    store_dis, load_dis, add_corr, sort_dis, calculate_kl, topk_dis, load_wiki_uni, \
    calculate_prompt_only_dis_kl_div, sum_dis
from models import build_model_wrapper
import sys


def all_data_evaluation(args):
    set_seed(0)

    args = get_relation_args(args)
    model_wrapper = build_model_wrapper(args.model_name, device=args.cuda_device)

    table = PrettyTable(
        field_names=["id", "label", "prompt", "lama", "wiki_uni", "lama_o", "wiki_uni_o"]
    )
    table.title = "{} {}".format(args.model_name, args.relation_type)

    lamadata = LamaDataset(relation_file=args.relation_file,
                           sample_dir=args.sample_dir,
                           sample_file_type=args.sample_file_type)
    id2relation, id2samples = lamadata.get_samples()

    print(len(id2relation))

    for relation_id in id2relation:

        relation = id2relation[relation_id]
        relation_label = relation["label"]
        relation_prompt = relation["template"]
        lama = id2samples[relation_id]
        lama_o = delete_overlap(lama)

        wiki_uni = load_wiki_uni(relation_id, model_prefix(args.model_name))
        wiki_uni_o = delete_overlap(wiki_uni)

        lama_p = model_wrapper.evaluate_samples(
            relation, lama,
            mask_pos=0, batch_size=args.batch_size, topk=args.topk, max_len=args.max_len
        )[1]
        wiki_uni_p = model_wrapper.evaluate_samples(
            relation, wiki_uni,
            mask_pos=0, batch_size=args.batch_size, topk=args.topk, max_len=args.max_len
        )[1]
        lama_o_p = model_wrapper.evaluate_samples(
            relation, lama_o,
            mask_pos=0, batch_size=args.batch_size, topk=args.topk, max_len=args.max_len
        )[1]
        wiki_uni_o_p = model_wrapper.evaluate_samples(
            relation, wiki_uni_o,
            mask_pos=0, batch_size=args.batch_size, topk=args.topk, max_len=args.max_len
        )[1]

        table.add_row([
            relation_id, relation_label, relation_prompt,
            lama_p, wiki_uni_p, lama_o_p, wiki_uni_o_p
        ])
    print(args)
    table = get_table_stat(table)
    print(table)
    return table


def store_all_distribution(args):
    set_seed(0)

    model_wrapper = build_model_wrapper(args.model_name, args.cuda_device)

    if model_prefix(args.model_name) == "bert":
        distribution_dir = "data/bert_data/distribution"
        relation_types = ['lama_original', 'lama_mine', "lama_auto"]
    elif model_prefix(args.model_name) == "roberta":
        distribution_dir = "data/roberta_data/distribution"
        relation_types = ['roberta_original', 'roberta_mine', "roberta_auto"]
    else:
        raise RuntimeError("wrong model")

    for relation_type in relation_types:
        args.relation_type = relation_type
        args = get_relation_args(args)

        lamadata = LamaDataset(relation_file=args.relation_file,
                               sample_dir=args.sample_dir,
                               sample_file_type=args.sample_file_type)
        id2relation, id2samples = lamadata.get_samples()

        for relation_id in id2relation:
            relation = id2relation[relation_id]
            lama = id2samples[relation_id]
            wiki_uni = load_wiki_uni(relation_id, model_prefix(args.model_name))

            mask_dis = get_mask_distribution(args, relation, model_wrapper)
            store_dis(distribution_dir, relation_type, "mask", relation_id, mask_dis)

            lama_dis = get_predict_distribution(
                args, relation, lama, model_wrapper
            )
            store_dis(distribution_dir, relation_type, "lama", relation_id, lama_dis)

            wiki_uni_dis = get_predict_distribution(
                args, relation, wiki_uni, model_wrapper
            )
            store_dis(distribution_dir, relation_type, "wiki_uni", relation_id, wiki_uni_dis)

            lama_obj_dis = get_obj_distribution(lama)
            store_dis(distribution_dir, relation_type, "lama_obj", relation_id, lama_obj_dis)

            wiki_uni_obj_dis = get_obj_distribution(wiki_uni)
            store_dis(distribution_dir, relation_type, "wiki_uni_obj",
                      relation_id, wiki_uni_obj_dis)


def plot_prompt(args):
    if model_prefix(args.model_name) == "bert":
        pic_dir = 'pics/paper_pic/bert/Prompt'
        dis_dir = "data/bert_data/distribution"
        relation_types = ['lama_original', 'lama_mine', "lama_auto"]
    elif model_prefix(args.model_name) == "roberta":
        pic_dir = 'pics/paper_pic/roberta/Prompt'
        dis_dir = "data/roberta_data/distribution"
        relation_types = ['roberta_original', 'roberta_mine', "roberta_auto"]
    else:
        raise RuntimeError("model error")

    mask_vs_uniform = {r"$T_{man}$": [], r"$T_{mine}$": [], r"$T_{auto}$": []}

    for relation_type in relation_types:
        args.relation_type = relation_type
        args = get_relation_args(args)

        lamadata = LamaDataset(relation_file=args.relation_file,
                               sample_dir=args.sample_dir,
                               sample_file_type=args.sample_file_type)
        id2relation, _ = lamadata.get_samples()

        corr_table = PrettyTable(
            field_names=["relation_id", "relation_label",
                         "mask_vs_uniform"]
        )

        corr_table.title = "prompt only dis corr {}".format(relation_type)

        for relation_id in id2relation:
            mask_dis = load_dis(dis_dir, relation_type, "mask", relation_id)
            uniform_dis = load_dis(dis_dir, relation_type, "wiki_uni", relation_id)

            mask_dis = topk_dis(mask_dis, 1000)

            mask_vs_uniform_corr = compare_distribution(mask_dis, uniform_dis)
            add_corr(mask_vs_uniform, relation_type, mask_vs_uniform_corr)
    set_plt()

    plt.figure(figsize=[8, 3.2])
    plt.tight_layout()
    draw_box_plot(mask_vs_uniform, pic_name="mask_vs_uniform", pic_dir=pic_dir)

    plt.show()


def cal_prompt_only_div(args):
    set_seed(0)

    if model_prefix(args.model_name) == "bert":
        dis_dir = "data/bert_data/distribution"
    elif model_prefix(args.model_name) == "roberta":
        dis_dir = "data/roberta_data/distribution"
    else:
        raise RuntimeError("model error")

    args = get_relation_args(args)

    lamadata = LamaDataset(relation_file=args.relation_file,
                           sample_dir=args.sample_dir,
                           sample_file_type=args.sample_file_type)
    id2relation, relation2samples = lamadata.get_samples()

    kl_div = 0

    for relation_id in id2relation:
        samples = relation2samples[relation_id]
        lama_obj_dis = get_obj_distribution(samples)
        mask_dis = load_dis(dis_dir, args.relation_type, "mask", relation_id)
        mask_vs_obj = calculate_kl(mask_dis, lama_obj_dis)
        kl_div += mask_vs_obj

    kl_div = round(kl_div / len(id2relation), 2)
    print("{} - {}".format(args.relation_type, kl_div))


def plot_predict_lama_vs_uniform(args):
    if model_prefix(args.model_name) == "bert":
        pic_dir = 'pics/paper_pic/bert/Prompt'
        dis_dir = "data/bert_data/distribution"
        relation_types = ['lama_original', 'lama_mine', "lama_auto"]
    elif model_prefix(args.model_name) == "roberta":
        pic_dir = 'pics/paper_pic/roberta/Prompt'
        dis_dir = "data/roberta_data/distribution"
        relation_types = ['roberta_original', 'roberta_mine', "roberta_auto"]
    else:
        raise RuntimeError("model error")

    lama_vs_uniform = {r"$T_{man}$": [], r"$T_{mine}$": [], r"$T_{auto}$": []}

    for relation_type in relation_types:
        args.relation_type = relation_type
        args = get_relation_args(args)

        lamadata = LamaDataset(relation_file=args.relation_file,
                               sample_dir=args.sample_dir,
                               sample_file_type=args.sample_file_type)
        id2relation, _ = lamadata.get_samples()

        table = PrettyTable(field_names=[
            "id", "label", "corr"
        ])
        table.title = "lama vs wiki-uni corr / {}".format(relation_type)

        for relation_id in id2relation:
            relation_label = id2relation[relation_id]["label"]

            lama_dis = load_dis(dis_dir, relation_type, "lama", relation_id)
            wiki_balance_dis = load_dis(dis_dir, relation_type, "wiki_uni", relation_id)

            lama_vs_uniform_corr = compare_distribution(
                lama_dis, wiki_balance_dis
            )
            add_corr(lama_vs_uniform, relation_type, lama_vs_uniform_corr)

    set_plt()
    plt.figure(figsize=[8, 3.2])
    plt.tight_layout()
    draw_box_plot(lama_vs_uniform, pic_name="lama_vs_uniform_predict", pic_dir=pic_dir)

    plt.show()


def get_dis_str(i, dis):
    if i >= len(dis):
        return "-"
    else:
        return "{}-{}".format(dis[i][0], dis[i][1])


def stat_uniform(args):
    if model_prefix(args.model_name) == "bert":
        dis_dir = "data/bert_data/distribution"
        relation_type = "lama_original"
    elif model_prefix(args.model_name) == "roberta":
        dis_dir = "data/roberta_data/distribution"
        relation_type = "roberta_original"
    else:
        raise RuntimeError("model error")

    args.relation_type = relation_type
    args = get_relation_args(args)

    lamadata = LamaDataset(relation_file=args.relation_file,
                           sample_dir=args.sample_dir,
                           sample_file_type=args.sample_file_type)
    id2relation, _ = lamadata.get_samples()

    fields = ["id", "label"]
    for k in range(1, 6):
        fields.append("lama_obj_{}".format(k))
        fields.append("uni_obj_{}".format(k))

    cover_table = PrettyTable(
        field_names=fields
    )
    cover_table.title = "topk obj coverage - {}".format(args.model_name)

    for relation_id in id2relation:
        relation_label = id2relation[relation_id]["label"]

        lama_dis = load_dis(dis_dir, relation_type, "lama_obj", relation_id)
        lama_sum = sum_dis(lama_dis)
        lama_dis = sort_dis(lama_dis)

        wiki_uni_dis = load_dis(dis_dir, relation_type, "wiki_uni_obj", relation_id)
        wiki_sum = sum_dis(wiki_uni_dis)
        wiki_uni_dis = sort_dis(wiki_uni_dis)

        new_row = [relation_id, relation_label]
        lama_top = 0
        wiki_top = 0
        for k in range(5):
            if k < len(lama_dis):
                lama_top += lama_dis[k][1]
            new_row.append(mean_round(lama_top, lama_sum))
            if k < len(wiki_uni_dis):
                wiki_top += wiki_uni_dis[k][1]
            new_row.append(mean_round(wiki_top, wiki_sum))

        cover_table.add_row(new_row)
    cover_table = get_table_stat(cover_table)
    print(cover_table)

    fields = ["id", "label"]
    for k in range(1, 6):
        fields.append("lama_{}".format(k))
        fields.append("uni_{}".format(k))

    cover_table = PrettyTable(
        field_names=fields
    )
    cover_table.title = "topk prediction coverage - {}".format(args.model_name)

    for relation_id in id2relation:
        relation_label = id2relation[relation_id]["label"]

        lama_dis = load_dis(dis_dir, relation_type, "lama", relation_id)
        lama_sum = sum_dis(lama_dis)
        lama_dis = sort_dis(lama_dis)

        wiki_uni_dis = load_dis(dis_dir, relation_type, "wiki_uni", relation_id)
        wiki_sum = sum_dis(wiki_uni_dis)
        wiki_uni_dis = sort_dis(wiki_uni_dis)

        new_row = [relation_id, relation_label]
        lama_top = 0
        wiki_top = 0
        for k in range(5):
            if k < len(lama_dis):
                lama_top += lama_dis[k][1]
            new_row.append(mean_round(lama_top, lama_sum))
            if k < len(wiki_uni_dis):
                wiki_top += wiki_uni_dis[k][1]
            new_row.append(mean_round(wiki_top, wiki_sum))

        cover_table.add_row(new_row)
    cover_table = get_table_stat(cover_table)
    print(cover_table)
