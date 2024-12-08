from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
