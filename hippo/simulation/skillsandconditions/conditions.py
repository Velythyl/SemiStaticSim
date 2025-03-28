from dataclasses import dataclass
from typing import Callable, List

from typing_extensions import Self

from hippo.simulation.ai2thor_metadata_reader import get_robot_inventory, get_object_list_from_controller
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

        acc = [] + self.prev.error_message()

        if self.state is False:

            acc = acc + [self._error_message()]

        acc = list(filter(lambda x: x is not None, acc))

        #acc = [Exception(a) for a in acc]

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

    def error_message(self) -> List[str]:
        return []

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
class COND_AuxiliaryObjectIsInInventory(Condition):
    name: str = "IsInInventory"

    def _error_message(self):
        return f"Object {self.sas.auxiliary_object.id} is not in the robot's inventory."

    def call(self, sas: SimulationActionState) -> bool:
        return sas.auxiliary_object.heldBy == f"robot{sas.robot+1}"

@dataclass
class COND_IsInInventory(Condition):
    name: str = "IsInInventory"

    def _error_message(self):
        return f"Object {self.sas.target_object.id} is not in the robot's inventory."

    def call(self, sas: SimulationActionState) -> bool:
        return sas.target_object.heldBy == f"robot{sas.robot+1}"

@dataclass
class _COND_AttributeEnabled(Condition):
    name: str = "AttributeEnabled"

    def _error_message(self):
        return f"The object {self.sas.target_object.id} does not support attribute {self.sas.skill_object.enabled_name}."

    def call(self, sas: SimulationActionState) -> bool:
        return bool(sas.skill_object.is_enabled)

@dataclass
class __COND_AttributeEnabledInAi2Thor(Condition):
    name: str = "AI2ThorAttributeEnabled"

    def _error_message(self):
        return f"The object {self.sas.target_object.id} does not support attribute {self.sas.skill_object.enabled_name}. This is likely due to a bug, and not a planning problem. Please report this bug."

    def call(self, sas: SimulationActionState) -> bool:
        obj = sas.get_object_from_controller(sas.target_object)
        return obj[sas.skill_object.enabled_name]

@dataclass
class COND_SkillEnabled(Condition):
    name: str = "SkillEnabled"
    prev: _Condition = _COND_AttributeEnabled()

    def _error_message(self):
        return f"The object {self.sas.target_object.id} does not support the skill {self.sas.skill_name}."

    def call(self, sas: SimulationActionState) -> bool:
        return sas.skill_object.has_skill_of_name(sas) and bool(sas.skill_object.is_enabled)

@dataclass
class COND_SlicingImplementInInventory(Condition):
    name: str = "SlicingImplementInInventory"
    prev: _Condition = _COND_AttributeEnabled()

    def _error_message(self):
        return f"The robot does not have access to a slicing implement."

    def call(self, sas: SimulationActionState) -> bool:
        inventory = get_robot_inventory(sas.controller, sas.robot)
        found_it = False
        for item in inventory:
            item = sas.pre_container.get_object_by_id(item)
            toolskill = item.skill_portfolio.find_skill("SlicingTool")
            if toolskill.is_enabled:
                found_it = True
                break
        return found_it

def verify_all_conditions(sas: SimulationActionState, condlist: List[_Condition]):
    ret = [c(sas) for c in condlist]
    return ret



if __name__ == "__main__":
    c = COND_SkillEnabled()
    i=0