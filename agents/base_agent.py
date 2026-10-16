from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """"Kelas dasar untuk semua agent dalam sistem."""

    def __init__(self, name: str):
        self.name = name


    @abstractmethod
    def run(self, data: Any) -> Any:
        """"Menjalankan proses utama agent."""
        raise NotImplementedError(
            f"Agent {self.name} harus mengimplementasikan run()."
        )