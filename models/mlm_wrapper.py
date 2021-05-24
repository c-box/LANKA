from utils.constant import CUDA_DEVICE, BATCH_SIZE
from utils.utils import get_pair
from transformers import PreTrainedTokenizer, PreTrainedModel, pipeline
from models.model_wrapper import ModelWrapper
import torch
from tqdm import tqdm


class MLMWrapper(ModelWrapper):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 model: PreTrainedModel,
                 device: int = None):
        super().__init__(tokenizer, model, device)

    # get the predict logits and position of mask token
    def get_predict_score(self,
                          input_text: list,
                          max_len=256
                          ):
        inputs = self.tokenizer.batch_encode_plus(
            input_text, padding="longest", truncation=True, max_length=max_len
        )

        mask_token_id = self.tokenizer.mask_token_id
        mask_index = []
        input_ids = inputs["input_ids"]
        # support multi mask
        for ids in input_ids:
            index = self.get_index(ids, mask_token_id)
            mask_index.append(index)

        for key in inputs:
            inputs[key] = torch.tensor(inputs[key]).cuda(self.device)
        outputs = self.model(**inputs, return_dict=True)
        predict_logits = outputs.logits

        return predict_logits, mask_index

    def logits_to_results(self, logits, topk):
        logits = torch.softmax(logits, dim=-1)
        predicted_prob, predicted_index = torch.topk(logits, topk)
        predicted_prob = predicted_prob.detach().cpu().numpy()
        predicted_index = predicted_index.cpu().numpy().tolist()
        predicted_tokens = []
        for index in predicted_index:
            predicted_tokens.append(self.tokenizer.decode([index]).strip())
        return predicted_tokens, predicted_prob

    # token to idx, for roberta and gpt, need to overload
    def token_to_idx(self, token):
        index = self.tokenizer.convert_tokens_to_ids(token)
        return index

    # return the rank and mrr
    def logits_to_results_with_obj(self, logits, topk, obj, rank_k=10000):
        predicted_tokens, predcited_probs = self.logits_to_results(logits, topk)
        logits = torch.softmax(logits, dim=-1)
        obj_index = self.token_to_idx(obj)
        obj_prob = logits[obj_index].item()

        rank_prob, rank_index = torch.topk(logits, rank_k)
        rank_index = rank_index.cpu().numpy().tolist()

        if obj_index not in rank_index:
            obj_rank = rank_k
            mrr = 0
        else:
            obj_rank = rank_index.index(obj_index) + 1
            mrr = 1 / obj_rank

        return predicted_tokens, predcited_probs, obj_prob, obj_rank, mrr

    # return the predict results given the input sentences as input
    def predict(self,
                input_texts: list,
                mask_pos=0,
                batch_size=BATCH_SIZE,
                obj_tokens=None,
                topk=10,
                rank_k=10000,
                max_len=256
                ):
        """
        :param input_texts:
        :param mask_pos: 0 for the fist mask, -1 for the last mask, list for particular assign
        :param batch_size:
        :param obj_tokens: if provide, will return the rank and prob info
        :param topk:
        :param rank_k:
        :param max_len:
        :return: predict_results
        """
        assert isinstance(mask_pos, int) or isinstance(mask_pos, list)
        if isinstance(mask_pos, int):
            mask_pos_lst = [mask_pos] * len(input_texts)
        else:
            mask_pos_lst = mask_pos
        assert len(mask_pos_lst) == len(input_texts)

        batch_text = self.partition(input_texts, batch_size)
        batch_mask_pos = self.partition(mask_pos_lst, batch_size)

        predict_results = []

        if obj_tokens is None:
            for idx in tqdm(range(len(batch_text))):
                single_batch_text = batch_text[idx]
                single_batch_mask_pos = batch_mask_pos[idx]

                predict_logits, mask_index = self.get_predict_score(
                    single_batch_text, max_len=max_len
                )

                for i in range(len(single_batch_text)):
                    assert isinstance(single_batch_mask_pos[i], int)
                    mask_pos_id = single_batch_mask_pos[i]
                    logits = predict_logits[i][mask_index[i][mask_pos_id]]

                    predicted_tokens, predicted_prob = self.logits_to_results(
                        logits, topk=topk
                    )
                    predict_results.append({'predict_tokens': predicted_tokens,
                                            'predict_prob': predicted_prob})
        else:
            assert len(obj_tokens) == len(input_texts)
            batch_obj = self.partition(obj_tokens, batch_size)

            for idx in tqdm(range(len(batch_text))):
                single_batch_text = batch_text[idx]
                single_batch_mask_pos = batch_mask_pos[idx]
                single_batch_obj = batch_obj[idx]

                predict_logits, mask_index = self.get_predict_score(
                    single_batch_text, max_len=max_len
                )

                for i in range(len(single_batch_text)):
                    assert isinstance(single_batch_mask_pos[i], int)
                    mask_pos_id = single_batch_mask_pos[i]
                    logits = predict_logits[i][mask_index[i][mask_pos_id]]
                    obj = single_batch_obj[i]

                    predicted_tokens, predicted_prob, obj_prob, obj_rank, mrr = \
                        self.logits_to_results_with_obj(
                            logits, topk, obj, rank_k=rank_k
                        )

                    if single_batch_obj[i] == predicted_tokens[0]:
                        predict_ans = True
                    else:
                        predict_ans = False

                    predict_results.append({'predict_tokens': predicted_tokens,
                                            'predict_prob': predicted_prob,
                                            'obj_prob': obj_prob,
                                            'obj_rank': obj_rank,
                                            'predict_ans': predict_ans,
                                            'mrr': mrr})
        return predict_results

    def prompt_to_sent(self, prompt: str, sub, obj):
        assert "[X]" in prompt
        assert "[Y]" in prompt
        sent = prompt.replace("[X]", sub)
        mask_token = self.tokenizer.mask_token
        sent = sent.replace("[Y]", mask_token)
        return sent

    # one mask only
    def evaluate_samples(self, relation, samples, pass_obj=False, **kwargs):
        relation_prompt = relation["template"]
        input_texts = []
        gold_obj = []
        p_1 = 0

        for sample in samples:
            sub, obj = get_pair(sample)
            gold_obj.append(obj)
            sent = self.prompt_to_sent(relation_prompt, sub, obj)
            input_texts.append(sent)

        if pass_obj:
            predict_results = self.predict(
                input_texts, obj_tokens=gold_obj, **kwargs
            )
        else:
            predict_results = self.predict(
                input_texts, **kwargs
            )

        for i in range(len(predict_results)):
            predict_token = predict_results[i]["predict_tokens"][0]
            if predict_token == gold_obj[i]:
                p_1 += 1

        if len(gold_obj) == 0:
            p_1 = 0
        else:
            p_1 = round(p_1 * 100 / len(gold_obj), 2)

        return predict_results, p_1
