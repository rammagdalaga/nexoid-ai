from collections import defaultdict
from typing import Any, Dict, List
from db.interface import DBInterface


class MemoryDB(DBInterface):
    def __init__(self):
        self._collections = defaultdict(list)
        self._indexes = defaultdict(set)

    def create_collection(self, name: str, indexes=None) -> None:
        _ = self._collections[name]
        if indexes:
            for idx in indexes:
                self._indexes[name].add(idx)

    def insert(self, collection: str, record: Dict[str, Any]) -> None:
        self._collections[collection].append(dict(record))

    def query(self, collection: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = self._collections.get(collection, [])
        out = []
        for row in rows:
            ok = True
            for k, v in filters.items():
                if row.get(k) != v:
                    ok = False
                    break
            if ok:
                out.append(dict(row))
        return out

    def upsert(self, collection: str, key_field: str, record: Dict[str, Any]) -> None:
        key = record.get(key_field)
        if key is None:
            self.insert(collection, record)
            return
        rows = self._collections[collection]
        for i, row in enumerate(rows):
            if row.get(key_field) == key:
                rows[i] = dict(record)
                return
        rows.append(dict(record))
