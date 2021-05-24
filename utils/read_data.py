import json
import os
from utils.utils import get_relations as get_lama_relation
from utils.utils import load_file, load_json_dic
from tqdm import tqdm


class LamaDataset:
    def __init__(self,
                 relation_file='data/relations_with_trigger.jsonl',
                 sample_dir='data/better_temp_res',
                 sample_file_type='json'):
        self.dataset = 'lama'
        self.relation_file = relation_file
        self.sample_dir = sample_dir
        self.sample_file_type = sample_file_type
        self.relations = get_lama_relation(self.relation_file)

        self.id2relation = {}
        self.relation2samples = {}
        for relation in self.relations:
            relation_id = relation['relation']

            if sample_file_type == 'json':
                file_name = '{}/{}'.format(self.sample_dir, relation_id)
                if os.path.isfile(file_name):
                    self.id2relation[relation_id] = relation
                    with open(file_name, 'r') as f:
                        samples = json.load(f)
                        self.relation2samples[relation_id] = samples
            else:
                file_name = '{}/{}.jsonl'.format(self.sample_dir, relation_id)
                # print(file_name)
                if os.path.isfile(file_name):
                    self.id2relation[relation_id] = relation
                    samples = load_file(file_name)
                    self.relation2samples[relation_id] = samples
            if os.path.isfile(file_name):
                pass

    def get_samples(self):
        return self.id2relation, self.relation2samples

    def get_combine_samples(self):
        combine_samples = []
        for relation_id in self.id2relation:
            samples = self.relation2samples[relation_id]
            combine_samples.extend(samples)
        return combine_samples
