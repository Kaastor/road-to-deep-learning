from typing import List

from pymongo import MongoClient


class MongoConnector:
    def __init__(self, database: str):
        uri = 'mongodb://127.0.0.1'
        self._mongo_client = MongoClient(uri)
        self.db = self._mongo_client.get_database(database)

    def query(self, query, collection) -> List:
        coll = self.db.get_collection(collection)
        return [e for e in coll.find(filter=query)]
