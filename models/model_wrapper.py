from utils.constant import CUDA_DEVICE, BATCH_SIZE
from transformers import PreTrainedTokenizer, PreTrainedModel


class ModelWrapper:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 model: PreTrainedModel,
                 device: int = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        # set the cuda device when init
        if device is None:
            device = CUDA_DEVICE
        self.device = device
        self.model.cuda(device)
        self.model.eval()

    def prompt_to_sent(self, prompt, sub, obj):
        pass

    @staticmethod
    def partition(ls: list, size: int):
        return [ls[i: i+size] for i in range(0, len(ls), size)]

    @staticmethod
    def get_index(ls, item):
        return [index for (index, value) in enumerate(ls) if value == item]
