import argparse
from prompt_based.prompt_eval import all_data_evaluation, plot_prompt, \
    store_all_distribution, plot_predict_lama_vs_uniform, stat_uniform, cal_prompt_only_div
from prompt_based.prompt_utils import check_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-type", type=str, default="lama_original",
                        choices=["lama_original", "lama_mine", "lama_auto",
                                 "roberta_original", "roberta_mine", "roberta_auto"])
    parser.add_argument("--model-name", type=str, default="bert-large-cased",
                        choices=["bert-large-cased", "roberta-large"])

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=10)

    parser.add_argument('--mask-topk', type=int, default=1000)

    parser.add_argument("--method", type=str, default="evaluation",
                        choices=[
                            "evaluation",
                            "prediction_corr",
                            "prompt_only_corr",
                            "store_all_distribution",
                            "topk_cover",
                            "cal_prompt_only_div",
                        ])

    args = parser.parse_args()
    check_args(args)

    if args.method == "evaluation":
        all_data_evaluation(args)
    elif args.method == "prediction_corr":
        plot_predict_lama_vs_uniform(args)
    elif args.method == "prompt_only_corr":
        plot_prompt(args)
    elif args.method == "store_all_distribution":
        store_all_distribution(args)
    elif args.method == "topk_cover":
        stat_uniform(args)
    elif args.method == "cal_prompt_only_div":
        cal_prompt_only_div(args)


if __name__ == '__main__':
    main()
