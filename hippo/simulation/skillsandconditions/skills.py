from dataclasses import dataclass

from hippo.simulation.skillsandconditions.conditions import COND_IsInProximity, COND_AttributeEnabledInAi2Thor, \
    COND_ObjectExists, COND_AttributeEnabled, COND_SkillEnabled
from hippo.simulation.skillsandconditions.sas import SimulationActionState
from hippo.simulation.skillsandconditions.skills_abstract import _Skill, get_exclusive_llm_names


@dataclass
class _ToggleObject(_Skill):
    enabled_name: str = "toggleable"
    state_name: str = "isToggled"

    @property
    def pre_conditions(self):
        return [COND_ObjectExists(), COND_IsInProximity(), COND_AttributeEnabled(), COND_SkillEnabled()]



@dataclass
class ToggleObjectOnAndOff(_ToggleObject):
    llm_name: str = "can be turned on/off"

    def ToggleObjectOn(self, sas: SimulationActionState):
        return self.replace(state_value=True)

    def ToggleObjectOff(self, sas: SimulationActionState):
        return self.replace(state_value=False)

@dataclass
class ToggleObjectOn(_ToggleObject):
    llm_name: str = "can only be turned on"

    def ToggleObjectOn(self, sas: SimulationActionState):
        return self.replace(state_value=True)

@dataclass
class ToggleObjectOff(_ToggleObject):
    llm_name: str = "can only be turned off"

    def ToggleObjectOff_(self, sas: SimulationActionState):
        return self.replace(state_value=True)

@dataclass
class _Ai2ThorSkill(_Skill):
    # movement skill, always supported
    state_value: bool = None    # must be inferred from the controller, so runtimeobject handles it...
    state_name: str = None

@dataclass
class GoToObject(_Ai2ThorSkill):
    # movement skill, always supported
    enabled_name: str = "navigable"
    is_enabled: bool = True # always enabled

    llm_name: str = None    # LLM cant turn on/off

    def GoToObject(self, sas: SimulationActionState):
        sas.action_callback()
        return None

    @property
    def pre_conditions(self):
        return [COND_ObjectExists()]

@dataclass
class PickupObject(_Ai2ThorSkill):
    # movement skill, always supported
    enabled_name: str = "pickupable"  # LLM CAN turn on/off

    llm_name: str = "can be picked up"

    @property
    def pre_conditions(self):
        return [COND_ObjectExists(), COND_IsInProximity(), COND_AttributeEnabled(),  COND_AttributeEnabledInAi2Thor(), COND_SkillEnabled()]

    def PickupObject(self, sas: SimulationActionState):
        sas.action_callback()
        return None

    def specific_post_conditions(self):
        return []# fixme make condition that checks that (1) item has isPickedUp and also that (2) robot has it in the inventory

@dataclass
class PutObject(_Ai2ThorSkill):
    enabled_name: str = "receptacle"  # LLM CAN turn on/off
    llm_name: str = "objects can be put down on this"

    def PutObject(self, sas: SimulationActionState):
        sas.action_callback()
        return None


    @property
    def pre_conditions(self):
        return [COND_ObjectExists(), COND_IsInProximity(), COND_AttributeEnabled(),  COND_AttributeEnabledInAi2Thor(), COND_SkillEnabled()]
    # fixme checks for the post conditions of PickupObject. So, the robot must have an object in its inventory.  Note: the target_object is the target surface, not the object that will be put down!

    @property
    def post_conditions(self):
        return [] # fixme  checks that the runtime container says the object that was in the inventory is now placed on the target object "isOnTop" on whatever


@dataclass
class _OpenableObject(_Skill):
    enabled_name: str = "openable"
    state_name: str = "isOpen"

    @property
    def pre_conditions(self):
        return [COND_ObjectExists(), COND_IsInProximity(), COND_AttributeEnabled(), COND_SkillEnabled()]


@dataclass
class OpenAndCloseObject(_OpenableObject):
    llm_name: str = "can be opened and closed"

    def OpenObject(self, sas: SimulationActionState):
        return self.replace(state_value=True)

    def CloseObject(self, sas: SimulationActionState):
        return self.replace(state_value=False)


@dataclass
class OpenObject(_OpenableObject):
    llm_name: str = "can only be opened"

    def OpenObject(self, sas: SimulationActionState):
        return self.replace(state_value=True)

@dataclass
class CloseObject(_OpenableObject):
    llm_name: str = "can only be closed"

    def CloseObject(self, sas: SimulationActionState):
        return self.replace(state_value=False)

@dataclass
class SliceObject(_Skill):
    enabled_name: str = "sliceable"
    state_name: str = "isSliced"

    llm_name: str = "can be sliced"

    def SliceObject(self, sas: SimulationActionState):
        return self.replace(state_value=True)

    @property
    def pre_conditions(self):
        return [COND_ObjectExists(), COND_IsInProximity(), COND_AttributeEnabled(),
                COND_SkillEnabled()]


@dataclass
class BreakObject(_Skill):
    enabled_name: str = "breakable"
    state_name: str = "isBroken"

    llm_name: str = "can be broken"

    @property
    def pre_conditions(self):
        return [COND_ObjectExists(), COND_IsInProximity(), COND_AttributeEnabled(),
                COND_SkillEnabled()]

    def BreakObject(self, sas: SimulationActionState):
        return self.replace(state_value=True)


if __name__ == "__main__":
    temp = get_exclusive_llm_names()
    i=0