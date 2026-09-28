from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class PublishError(Exception):
    transient = True


class TransientPublishError(PublishError):
    transient = True


class PermanentPublishError(PublishError):
    transient = False


@dataclass(frozen=True)
class PublishItem:
    photo_id: int
    path: Path
    sha256: str
    source_name: str


class Publisher(ABC):
    name = "base"

    @abstractmethod
    def publish(self, item: PublishItem) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    def close(self) -> None:
        return None
