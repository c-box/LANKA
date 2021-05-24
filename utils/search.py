import argparse
import json
import string
from utils.elasticsearch_client import ElasticsearchClient
from typing import List, Dict
import queue
from tqdm import tqdm
from utils.read_data import LamaDataset
from utils.utils import get_relation_args


def generate_search_body(args, sub, obj):
    search_body = {
        "query": {
            "bool": {
                "must": [
                    {"match_phrase": {"sentence": sub}},
                    {"match_phrase": {"sentence": obj}},
                ]
            }
        }
    }
    return search_body


def search_word(args, keyword: str, es: ElasticsearchClient):
    search_body = {
        "query": {
            "match_phrase": {
                "sentence": keyword
            }
        }
    }
    response = es.search(body=search_body, index=args.index, scroll='1m')
    cnt = response['hits']['total']['value']
    return cnt


def search_by_kerwords(args, kerwords, es: ElasticsearchClient, index=None):
    query = []
    for keyword in kerwords:
        query.append({"match_phrase": {"sentence": keyword}})

    search_body = {
        "query": {
            "bool": {
                "must": query
            }
        }
    }
    if index is None:
        response = es.search(body=search_body, index=args.index, scroll='1m')
    else:
        response = es.search(body=search_body, index=index, scroll='1m')
    cnt = response['hits']['total']['value']
    return cnt


def search_by_mention(args, es: ElasticsearchClient, triple: Dict, res_type='count'):
    sub = triple['sub']
    obj = triple['obj']
    search_body = generate_search_body(args, sub, obj)
    response = es.search(body=search_body, index=args.index)
    if res_type == 'count':
        cnt = response['hits']['total']['value']
        return cnt
    else:
        search_sentences = []
        for hit in response['hits']['hits']:
            search_sentences.append(hit['_source']['sentence'])
        if res_type == 'sentence':
            return search_sentences
        elif res_type == 'template':
            search_templates = []
            for sentence in search_sentences:
                template = sentence.replace(sub, '[X]', 1)
                template = template.replace(obj, '[Y]', 1)
                search_templates.append(template)
            return search_templates


def top_sort(graph: Dict):
    in_degrees = dict((u, 0) for u in graph)
    depth = dict((u, 1) for u in graph)
    for u in graph:
        for v in graph[u]:
            in_degrees[v] += 1
    que = [u for u in in_degrees if in_degrees[u] == 0]
    seq = []
    for u in que:
        depth[u] = 1
    while que:
        u = que.pop()
        seq.append(u)
        for v in graph[u]:
            in_degrees[v] -= 1
            depth[v] = max(depth[v], depth[u] + 1)
            if in_degrees[v] == 0:
                que.append(v)
    in_degrees = sorted(in_degrees.items(), key=lambda x: depth[x[0]])
    for u in in_degrees:
        if u[1] > 0:
            seq.append(u[0])

    assert len(seq) == len(graph)
    return seq, depth


def bfs_class_set(class_set, es, index='wikidata_tuples_p279'):
    vis = set()
    que = queue.Queue()
    for subclass in class_set:
        que.put(subclass)
        vis.add(subclass['id'])
    while not que.empty():
        head_class = que.get()
        head_id = head_class['id']
        search_body = {
            "query": {
                "match": {
                    "sub_id": head_id
                }
            }
        }
        response = es.search(body=search_body, index=index, scroll='1m', size=100)
        hits = response['hits']['hits']
        hits = deduplication(hits)
        for hit in hits:
            hit = hit['_source']
            tail_class = {"id": hit['obj_id'], "label": hit['obj_label'], "description": hit['obj_description']}

            if tail_class['id'] not in vis:
                class_set.append(tail_class)
                que.put(class_set[-1])
                vis.add(tail_class['id'])

    return class_set


def get_sample_class(es: ElasticsearchClient, entity_id=None, entity_name=None, index='wikidata_tuples_p31', depth=-1):
    if entity_id is not None:
        search_body = {
            "query": {
                "match": {
                    "sub_id": entity_id
                }
            }
        }
    elif entity_name is not  None:
        search_body = {
            "query": {
                "match": {
                    "sub_label": entity_name
                }
            }
        }
    else:
        raise RuntimeError("Not enough information")

    responce = es.search(body=search_body, index=index, scroll='1m', size=100)
    hits = responce['hits']['hits']
    hits = deduplication(hits)
    class_set = []
    for hit in hits:
        hit = hit['_source']
        class_set.append({"id": hit['obj_id'], "label": hit['obj_label'], "description": hit['obj_description'],
                          'depth': 1})
    class_set = bfs_class_set(class_set, es)
    if len(class_set) == 0:
        print(entity_id)
        return class_set
    max_depth = max(class_set, key=lambda x: x['depth'])['depth']
    for class_label in class_set:
        class_label['depth'] = max_depth - class_label['depth']

    if depth != -1:
        class_set = [x for x in class_set if x['depth'] >= depth]
    return class_set


def get_samples_class_distribution(es: ElasticsearchClient, entity_names=None, entity_ids=None,
                                   threshold=0.8, class_threshold=0.9):
    class_sets = []
    if entity_ids is not None:
        for entity_id in tqdm(entity_ids):
            class_set = get_sample_class(es, entity_id=entity_id)
            class_sets.append(class_set)
        sample_num = len(entity_ids)
    elif entity_names is not None:
        for entity_name in tqdm(entity_names):
            class_set = get_sample_class(es, entity_name=entity_name)
            class_sets.append(class_set)
        sample_num = len(entity_names)
    else:
        raise RuntimeError("Not enough information")

    class_distribution = {}
    for class_set in class_sets:
        for class_label in class_set:
            class_id = class_label['id']
            if class_id not in class_distribution:
                class_distribution[class_id] = {}
                class_distribution[class_id]['label'] = class_label
                class_distribution[class_id]['frequency'] = 0
            class_distribution[class_id]['frequency'] += 1
    class_distribution = sorted(class_distribution.items(), key=lambda x: x[1]['label']['depth'])
    filter_distribution = []
    depth = 0
    max_frequency = 0
    temp_distribution = []
    sample_class = []

    for dis in class_distribution:
        if dis[1]['label']['depth'] == depth:
            temp_distribution.append(dis)
            max_frequency = max(max_frequency, dis[1]['frequency'])
        else:
            if max_frequency >= class_threshold * sample_num:
                sample_class = []
                for temp_dis in temp_distribution:
                    if temp_dis[1]['frequency'] >= threshold * max_frequency:
                        sample_class.append(temp_dis[1]['label'])

            depth = dis[1]['label']['depth']
            filter_distribution = filter_distribution + [x for x in temp_distribution
                                                         if x[1]['frequency'] >= threshold * max_frequency]
            max_frequency = dis[1]['frequency']
            temp_distribution = [dis]
    return filter_distribution, sample_class


def update_distribution_frequency(class_distribution, class_id, class_graph, vis):
    class_distribution[class_id]['frequency'] += 1
    for v in class_graph[class_id]:
        if v in vis:
            continue
        vis.add(v)
        update_distribution_frequency(class_distribution, v, class_graph, vis)


def update_distribution(class_distribution, class_id, class_graph, que, vis, subclass):
    vis.add(class_id)
    if class_id in class_distribution:
        update_distribution_frequency(class_distribution, class_id, class_graph, vis)
    else:
        que.put(subclass)
        class_distribution[class_id] = {"label": subclass, "frequency": 1}
        class_graph[class_id] = []


def bfs_class_set_with_graph(class_set, es, class_graph, class_distribution, index='wikidata_tuples_p279'):
    que = queue.Queue()
    vis = set()
    for subclass in class_set:
        class_id = subclass['id']
        if class_id not in vis:
            update_distribution(class_distribution, class_id, class_graph, que, vis, subclass)

    while not que.empty():
        head_class = que.get()
        head_id = head_class['id']
        search_body = {
            "query": {
                "match": {
                    "sub_id": head_id
                }
            }
        }
        response = es.search(body=search_body, index=index, scroll='1m', size=100)
        hits = response['hits']['hits']
        hits = deduplication(hits)
        for hit in hits:
            hit = hit['_source']
            tail_class = {"id": hit['obj_id'], "label": hit['obj_label'],
                          "description": hit['obj_description']}
            tail_id = tail_class['id']
            if tail_id not in class_graph[head_id]:
                class_graph[head_id].append(tail_id)
            if tail_class['id'] not in vis:
                update_distribution(class_distribution, tail_id, class_graph, que, vis, tail_class)

    return class_distribution, class_graph


def search_for_class(es: ElasticsearchClient, entity_id=None, entity_name=None,
                     index='wikidata_tuples_p31'):
    if entity_id is not None:
        search_body = {
            "query": {
                "match": {
                    "sub_id": entity_id
                }
            }
        }
    elif entity_name is not  None:
        search_body = {
            "query": {
                "match": {
                    "sub_label": entity_name
                }
            }
        }
    else:
        raise RuntimeError("Not enough information")
    responce = es.search(body=search_body, index=index, scroll='1m', size=100)
    class_set = []
    hits = responce['hits']['hits']
    hits = deduplication(hits)
    for hit in hits:
        hit = hit['_source']
        class_set.append({"id": hit['obj_id'], "label": hit['obj_label'],
                          "description": hit['obj_description']})
    return class_set


def deduplication(raw_hits):
    hits = []
    for hit in raw_hits:
        idx = hit['_source']["obj_id"]
        flag = True
        for new_hit in hits:
            if idx == new_hit['_source']["obj_id"]:
                flag = False
                break
        if flag:
            hits.append(hit)

    return hits


def get_sample_class_distribution_with_graph(es: ElasticsearchClient,
                                             entity_names=None, entity_ids=None, threshold=0.85):
    class_distribution = {}
    class_graph = {}
    sample_num = 0
    if entity_ids is not None:
        for entity_id in tqdm(entity_ids):
            p31_class_set = search_for_class(es, entity_id=entity_id, index="wikidata_tuples_p31")
            p279_class_set = search_for_class(es, entity_id=entity_id, index="wikidata_tuples_p279")
            class_set = p31_class_set + p279_class_set
            if len(class_set) > 0:
                sample_num += 1
            class_distribution, class_graph = \
                bfs_class_set_with_graph(class_set, es, class_graph, class_distribution)
            for dis in class_distribution:
                if class_distribution[dis]['frequency'] > sample_num:
                    print(sample_num)
    elif entity_names is not None:
        for entity_name in tqdm(entity_names):
            class_set = get_sample_class(es, entity_name=entity_name)
            if len(class_set) > 0:
                sample_num += 1
            class_distribution, class_graph = \
                bfs_class_set_with_graph(class_set, es, class_graph, class_distribution)
    else:
        raise RuntimeError("Not enough information")
    tops_sort_seq, depth = top_sort(class_graph)
    res_class = {}
    for i in range(len(tops_sort_seq)):
        idx = tops_sort_seq[i]
        if class_distribution[idx]['frequency'] >= threshold * sample_num:
            res_class[idx] = class_distribution[idx]
            for j in range(i+1, len(tops_sort_seq)):
                j_idx = tops_sort_seq[j]
                if depth[j_idx] == depth[idx]:
                    res_class[j_idx] = class_distribution[j_idx]
                else:
                    break
            break
    return res_class


def relation_to_wikidata_samples(es: ElasticsearchClient, relation_id, index='wikidata_tuples'):
    search_body = {
        "query": {
            "match": {
                "relation_id": relation_id
            }
        },
        "size": 1000
    }
    responce, scroll_id = es.scroll_search(body=search_body, index=index, scroll='1m')
    num_of_samples = responce['hits']['total']['value']
    print("num of samples: {}".format(num_of_samples))
    samples = []
    while True:
        for hit in responce['hits']['hits']:
            hit = hit['_source']
            sample = {"sub_uri": hit['sub_id'], "sub_label": hit['sub_label'],
                      "sub_description": hit['sub_description'],
                      "obj_uri": hit['obj_id'], "obj_label": hit['obj_label'],
                      "obj_description": hit['obj_description']}
            samples.append({'sample': sample})
        responce = es.scroll(scroll_id=scroll_id)
        if len(responce['hits']['hits']) == 0:
            break
    return samples


def search_sample_type(es, samples, search_arg='obj_uri'):
    entity_ids = []
    for sample in samples:
        entity_ids.append(sample['sample'][search_arg])
    res_class = get_sample_class_distribution_with_graph(es, entity_ids=entity_ids)
    return res_class


def main():
    es = ElasticsearchClient()
    es.check_connection()
    parser = argparse.ArgumentParser()
    parser.add_argument('--relation-type', type=str, default="lama_original")
    args = parser.parse_args()
    args = get_relation_args(args)
    lama_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)
    id2relation, relation2samples = lama_data.get_samples()
    relation_id = "P1303"
    samples = relation2samples[relation_id]
    res = search_sample_type(es, samples)
    print(res)


if __name__ == '__main__':
    main()
