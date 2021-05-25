from utils.utils import get_pair, model_prefix
from utils.mlm_predict import count_sub_words
import re
from prettytable import PrettyTable
from models.mlm_wrapper import MLMWrapper


def mask_obj(context, obj, mask_mark):
    return re.sub(re.escape(obj), re.escape(mask_mark), context)


def mask_obj_as_whole(context, obj, mask_token, all_mask=False):
    context = context.split(" ")
    if all_mask is False:
        flag = False
        for i in range(len(context)-1, -1, -1):
            token = context[i]
            if obj in token:
                context[i] = mask_token
                flag = True
                break
    else:
        for i in range(len(context)-1, -1, -1):
            token = context[i]
            if obj in token:
                context[i] = mask_token
    return " ".join(context)


def mask_sub(context, sub, mask_mark):
    try:
        ans = re.sub(re.escape(sub), re.escape(mask_mark), context)
        return ans
    except:
        print(context)
        print(sub)
        raise RuntimeError("error")


def get_new_context(args, sample, template=None, tokenizer=None, mask_token="[MASK]"):
    if "drqa" in args.context_method:
        if "context" not in sample or len(sample["context"]) == 0:
            return ""
        context = sample['context'][0]
    else:
        context = sample["oracle_context"][0]
    if args.mask_obj:
        sub, obj = get_pair(sample)
        context = mask_obj_as_whole(context, obj, mask_token, all_mask=args.all_obj_mask)
        if count_sub_words(context, tokenizer) > 256:
            tokens = context.split()
            idx = 0
            for i in range(len(tokens)):
                token = tokens[i]
                if mask_token in token:
                    idx = i
                    break
            context = " ".join(tokens[max(0, idx - 50):])
    return context


def stat_context(context, sub, obj):
    if sub in context:
        if obj in context:
            return 0
        else:
            return 1
    elif obj in context:
        return 2
    else:
        return 3


def stat_context_by_obj(context, obj):
    if obj in context:
        return True
    else:
        return False


# 加context之后的预测结果
def get_result_with_context(args, relation, relation_samples,
                            model_wrapper: MLMWrapper,
                            no_context=False,  # 需要
                            context_only=False,
                            return_samples=False,
                            return_ans=False,
                            print_tokens=False,
                            contexts=None,
                            return_tokens=False,
                            return_obj_in_context=False):
    relation_template = relation['template']
    input_sentences = []
    gold_obj = []
    p_1 = 0
    p_10 = 0
    obj_in_contexts = []
    for i in range(len(relation_samples)):
        relation_sample = relation_samples[i]
        sub, obj = get_pair(relation_sample)
        gold_obj.append(obj)
        input_sentence = model_wrapper.prompt_to_sent(
            relation_template, sub, obj
        )

        if contexts is None:
            context = get_new_context(
                args, relation_sample,
                relation_template, tokenizer=model_wrapper.tokenizer,
                mask_token=model_wrapper.tokenizer.mask_token
            )
        else:
            context = contexts[i]

        if obj in context:
            obj_in_contexts.append(True)
        else:
            obj_in_contexts.append(False)

        if context_only:
            if args.mask_obj is False:
                raise RuntimeError("contradictory!")
            input_sentences.append(context)
        else:
            if no_context:
                input_sentences.append(input_sentence)
            else:
                if model_prefix(args.model_name) == "roberta":
                    input_sentence = input_sentence + '{}'.format(model_wrapper.tokenizer.sep_token) + context
                else:
                    input_sentence = [input_sentence, context]
                input_sentences.append(input_sentence)

    predict_results = model_wrapper.predict(
        input_texts=input_sentences,
        mask_pos=0,
        batch_size=args.batch_size,
        max_len=args.max_len
    )

    right_samples = []
    false_sample = []
    predict_ans = []

    all_tokens = []

    for i in range(len(predict_results)):
        predict_tokens = predict_results[i]['predict_tokens']
        predict_prob = predict_results[i]['predict_prob']
        sample = relation_samples[i]
        sub, obj = get_pair(sample)
        obj = gold_obj[i]
        if obj == predict_tokens[0]:
            p_1 += 1
            predict_ans.append(True)
        else:
            predict_ans.append(False)

        if obj in predict_tokens[: args.context_topk]:
            right_samples.append(relation_samples[i])
        else:
            false_sample.append(relation_samples[i])

        if obj in predict_tokens[: 10]:
            p_10 += 1

        if print_tokens:
            table = PrettyTable(field_names=["token"])
            table.title = "{}-{}".format(sub, obj)
            for token in predict_tokens[:10]:
                table.add_row([token])
            print(table)
        all_tokens.append(predict_tokens[:10])

    # print(predict_results)
    if len(gold_obj) == 0:
        p_1 = 0
    else:
        p_1 = round(p_1 * 100 / len(gold_obj), 2)

    # 只要precision
    if return_samples:
        return p_1, right_samples, false_sample
    else:
        if return_ans:
            if return_obj_in_context:
                return p_1, predict_ans, obj_in_contexts
            return p_1, predict_ans
        elif return_tokens:
            return p_1, all_tokens
        else:
            return p_1

