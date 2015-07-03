from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class BaseItertool(six.Iterator):
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass
