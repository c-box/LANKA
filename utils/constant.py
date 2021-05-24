CUDA_DEVICE = 3
BATCH_SIZE = 128

RELATION_FILES = {
    "lama_original": {
        "relation_file": "data/relations/relations_with_trigger.jsonl",
        "sample_dir": "data/bert_data/lama",
        "sample_file_type": "jsonl"
    },
    "lama_mine": {
        "relation_file": "data/relations/mine_relations_with_trigger.jsonl",
        "sample_dir": "data/bert_data/lama",
        "sample_file_type": "jsonl"
    },
    "lama_auto": {
        "relation_file": "data/relations/auto_relations_with_trigger.jsonl",
        "sample_dir": "data/bert_data/lama",
        "sample_file_type": "jsonl"
    },
    "lama_orginal_with_lama_drqa": {
        "relation_file": "data/relations_with_trigger.jsonl",
        "sample_dir": "data/bert_data/context_lama_drqa",
        "sample_file_type": "json"
    },

    "roberta_auto": {
        "relation_file": "data/relations/roberta_auto_relations_with_trigger.jsonl",
        "sample_dir": "data/roberta_data/lama",
        "sample_file_type": "json"
    },
    "roberta_original": {
        "relation_file": "data/relations/relations_with_trigger.jsonl",
        "sample_dir": "data/roberta_data/lama",
        "sample_file_type": "json"
    },
    "roberta_mine": {
        "relation_file": "data/relations/mine_relations_with_trigger.jsonl",
        "sample_dir": "data/roberta_data/lama",
        "sample_file_type": "json"
    },
}
