"""Abstract model test module"""
import abc
import typing


class AbstractModelTest(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model, model_specs_dict: typing.Dict, output_path: str):
        pass

    @abc.abstractmethod
    def execute(self, data):
        pass
