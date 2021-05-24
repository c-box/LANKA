import argparse
from context_based.implicit_leak import implicit_leak
from context_based.explicit_leak import explicit_leak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relation-type', type=str,
                        default="lama_orginal_with_lama_drqa",
                        choices=["lama_original",
                                 "lama_orginal_with_lama_drqa"])
    parser.add_argument('--model-name', type=str, default='bert-large-cased',
                        choices=["bert-large-cased", "roberta-large"])
    parser.add_argument('--context-method', type=str, default='lama_drqa',
                        choices=["lama_drqa"])

    parser.add_argument('--mask-obj', action='store_true')
    parser.add_argument('--mask-sub', action='store_true')
    parser.add_argument('--stat-context', action='store_true')
    parser.add_argument('--token-type', action='store_true')
    parser.add_argument('--context-topk', type=int, default=1)
    parser.add_argument('--false-only', action="store_true")
    parser.add_argument('--obj-only', action="store_true")

    parser.add_argument("--all-obj-mask", action="store_false")

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=256)

    parser.add_argument("--method", type=str, default="implicit_leak",
                        choices=["explicit_leak", "implicit_leak"])

    args = parser.parse_args()

    if args.method == "explicit_leak":
        explicit_leak(args)
    elif args.method == "implicit_leak":
        args.mask_obj = True
        implicit_leak(args)


if __name__ == '__main__':
    main()
