import argparse
from case_based.evaluate_analogy_resoning import evaluate_analogy_reasoning
from case_based.analogy_resoning_type import type_precision, type_rank_change


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-type", type=str, default="lama_original",
                        choices=["lama_original", "roberta_original"])

    parser.add_argument('--model-name', type=str, default='bert-large-cased',
                        choices=["bert-large-cased", "roberta-large"])
    parser.add_argument('--choose-method', type=str, default='random_without')
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10000)

    parser.add_argument('--few-shot-prompt', type=str, default='original')
    parser.add_argument('--sample-prompt', type=str, default='original',
                        choices=["original"])

    parser.add_argument('--num-of-shot', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--type-inc-k', type=int, default=10)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument('--task', type=str, default='evaluate_analogy_reasoning')
    args = parser.parse_args()

    if args.task == 'evaluate_analogy_reasoning':
        args.topk = 10
        evaluate_analogy_reasoning(args)
    elif args.task == "type_precision":
        type_precision(args)
    elif args.task == "type_rank_change":
        type_rank_change(args)


if __name__ == '__main__':
    main()
