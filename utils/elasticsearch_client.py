# -*- coding: utf-8 -*-

import logging
import os
from time import sleep

from elasticsearch import Elasticsearch, helpers

logging.basicConfig(format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                    filename='info.log', level=logging.INFO)
logger = logging.getLogger(__name__)


class ElasticsearchClient(object):
    def __init__(
        self, hosts=[{"host": os.environ.get("ES_HOST", "localhost"), "port": 9201}]
    ):
        self._client = Elasticsearch(hosts=hosts)

    def check_connection(self):
        logger.info("Connecting to ES")

        attemps = 0
        while not self._client.ping():
            logger.info("Connection Failed, Retrying...")
            attemps += 1
            if attemps > 5:
                return
            sleep(1)

        logger.info("Connected to ES.")

    def reset_index(self, index, body=None):
        logger.info("Rebulding index...")
        self._client.indices.delete(index=index, ignore=[400, 404])
        return self._client.indices.create(index=index, body=body, ignore=400)

    def create(self, data, index, doc_type, **kwargs):
        return self._client.index(
            index=index, doc_type=doc_type, body=data, request_tiemout=3600, ignore=400
        )

    def bulk(self, actions, index, doc_type, **kwargs):
        for success, info in helpers.parallel_bulk(
            self._client, actions, index=index, doc_type=doc_type, **kwargs
        ):
            if not success:
                logger.error("This document failed:", info)

    def search(self, body, index, **kwargs):
        ans = self._client.search(index=index, body=body, **kwargs)
        if '_scroll_id' in ans:
            self._client.clear_scroll(scroll_id=ans['_scroll_id'])
        return ans

    def scroll_search(self, body, index, scroll, **kwargs):
        ans = self._client.search(index=index, body=body, scroll=scroll, **kwargs)
        scroll_id = ans['_scroll_id']
        return ans, scroll_id

    def scroll(self, scroll_id):
        ans = self._client.scroll(scroll_id=scroll_id, scroll='1m')
        return ans


def main():
    es = ElasticsearchClient()
    es.check_connection()


if __name__ == "__main__":
    main()
