from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DBInterface(ABC):
    @abstractmethod
    def create_collection(self, name: str, indexes: Optional[List[str]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(self, collection: str, record: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, collection: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, collection: str, key_field: str, record: Dict[str, Any]) -> None:
        raise NotImplementedError
