from __future__ import annotations

from abc import ABC, abstractmethod

from research_flow.models import PaperRecord, QueryGroup


class BaseRetriever(ABC):
    source_name: str

    @abstractmethod
    def retrieve(self, query: QueryGroup, limit: int) -> list[PaperRecord]:
        raise NotImplementedError

