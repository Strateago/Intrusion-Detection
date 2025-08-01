"""Abstract strategy module"""
import abc
import typing


class AbstractModelTrainValidate(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model, model_config_dict: typing.Dict, output_path: str):
        pass

    @abc.abstractmethod
    def execute(self, train_data):
        pass
