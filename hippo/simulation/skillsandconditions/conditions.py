from dataclasses import dataclass
from typing import Callable, List

from typing_extensions import Self

from hippo.simulation.skillsandconditions.sas import SimulationActionState
from hippo.utils.selfdataclass import SelfDataclass


@dataclass
class _Condition(Callable, SelfDataclass):
    name: str = ""
    state: bool = None
    sas: SimulationActionState = None

    prev: Self = None
    success: bool = None

    def __bool__(self):
        return self.success

    def error_message(self) -> List[str]:
        if self.state is None:
            raise AssertionError("You tried to get the Condition's error message, but it has not been evaluated yet.")

        if self.state is True:
            return []

        acc = [] + self.prev.error_message()

        acc = acc + [self._error_message()]

        acc = list(filter(lambda x: x is not None, acc))

        return acc

    def _error_message(self):
        raise NotImplementedError()

    def __call__(self, sas: SimulationActionState) -> Self:
        return self


@dataclass
class TrivialCondition(_Condition):
    name: str = ""
    state: bool = True
    sas: SimulationActionState = None

    prev: Callable = None
    success: bool = True


    def __call__(self, sas: SimulationActionState) -> Self:
        return self.replace(sas=sas)

    def _error_message(self):
        return None


@dataclass
class Condition(_Condition):
    name: str
    state: bool = None
    sas: SimulationActionState = None

    prev: _Condition = TrivialCondition()
    success: bool = None

    def __call__(self, sas: SimulationActionState) -> Self:
        self = self.replace(sas=sas)
        if self.prev is not None:
            prev = self.prev(sas)
            self = self.replace(prev=prev)

        state = self.call(sas)
        self = self.replace(state=state)
        self = self.replace(success=self.state and self.prev.success)
        return self

    def call(self, sas: SimulationActionState) -> bool:
        # sets self.state to True/False
        raise NotImplementedError()

@dataclass
class COND_ObjectExists(Condition):
    name: str = "ObjectExists"

    def _error_message(self):
        return f"Object {self.sas.target_object.id} does not exist."

    def call(self, sas: SimulationActionState) -> bool:
        if sas.get_object_from_controller(sas.target_object) is None:
            return False
        return True

@dataclass
class COND_IsInProximity(Condition):
    name: str = "IsInteractable"
    prev: _Condition = COND_ObjectExists()

    def _error_message(self):
        return f"Object {self.sas.target_object.id} is not in proximity to the robot."

    def call(self, sas: SimulationActionState) -> bool:
        object = sas.get_object_from_controller(sas.target_object)
        return object["isInteractable"]

@dataclass
class COND_AttributeEnabled(Condition):
    name: str = "AttributeEnabled"

    def _error_message(self):
        return f"The object {self.sas.target_object.id} does not support attribute {self.sas.skill_object.enabled_name}."

    def call(self, sas: SimulationActionState) -> bool:
        return sas.skill_object.is_enabled

@dataclass
class COND_AttributeEnabledInAi2Thor(Condition):
    name: str = "AttributeEnabled"

    def _error_message(self):
        return f"The object {self.sas.target_object.id} does not support attribute {self.sas.skill_object.enabled_name}."

    def call(self, sas: SimulationActionState) -> bool:
        obj = sas.get_object_from_controller(sas.target_object)
        return obj[sas.skill_object.enabled_name]

@dataclass
class COND_SkillEnabled(Condition):
    name: str = "SkillEnabled"
    prev: _Condition = COND_AttributeEnabled()

    def _error_message(self):
        return f"The object {self.sas.target_object.id} cannot handle {self.sas.skill_name}."

    def call(self, sas: SimulationActionState) -> bool:
        return sas.skill_object.has_skill_of_name(sas)

def verify_all_conditions(sas: SimulationActionState, condlist: List[_Condition]):
    ret = [c(sas) for c in condlist]
    return ret



if __name__ == "__main__":
    c = COND_SkillEnabled()
    i=0