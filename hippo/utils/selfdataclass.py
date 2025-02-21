import dataclasses

@dataclasses.dataclass
class SelfDataclass:

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def asdict(self):
        return dataclasses.asdict(self)

    def astuple(self):
        return dataclasses.astuple(self)



