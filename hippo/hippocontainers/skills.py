from dataclasses import dataclass


class ObjectAlreadyBroken(AssertionError):
    pass


class _SkillUnsupprted(AssertionError):
    pass


class ObjectNotBreakable(_SkillUnsupprted):
    pass


class ObjectNotSliceable(_SkillUnsupprted):
    pass


class ObjectNotToggleable(_SkillUnsupprted):
    pass


class ObjectAlreadySliced(AssertionError):
    pass


class ObjectDoesNotExist(AssertionError):
    pass


@dataclass
class _Skill:
    enabled_name: str
    state_name: str

@dataclass
class Toggleable(_Skill):
    enabled_name: str = "toggleable"
    state_name: str = "isToggled"

class Breakable(_Skill):
    enabled_name: str = "breakable"
    state_name: str = "isBroken"

class Sliceable(_Skill):
    enabled_name: str = "sliceable"
    state_name: str = "isSliced"