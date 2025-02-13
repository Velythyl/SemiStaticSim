import dataclasses
from functools import wraps


def selfdataclass(cls):
    # Convert the class to a dataclass
    cls = dataclasses.dataclass(cls)

    # Make the class inherit from MyDataclass
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not isinstance(self, SelfDataclass):
            # Dynamically make this class inherit from MyDataclass
            self.__class__ = type(cls.__name__, (SelfDataclass, cls), {})

    cls.__init__ = new_init
    return cls

@dataclasses.dataclass
class SelfDataclass:

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def asdict(self):
        return dataclasses.asdict(self)

    def astuple(self):
        return dataclasses.astuple(self)



