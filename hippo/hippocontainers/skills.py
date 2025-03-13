import copy
import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

from hippo.hippocontainers.conditions import COND_ObjectExists, COND_AttributeEnabled, \
    COND_IsInProximity, COND_SkillEnabled, verify_all_conditions, COND_AttributeEnabledInAi2Thor
from hippo.hippocontainers.sas import SimulationActionState
from hippo.utils.selfdataclass import SelfDataclass


class _MultiException(Exception):
    def __init__(self, exceptions):
        super().__init__()
        self.exceptions = exceptions

class PreconditionFailed(_MultiException):
    pass

class PostconditionFailed(_MultiException):
    pass

def maybe_raise_many_exceptions(e_list, type: _MultiException):
    e_list = list(filter(lambda e: isinstance(e, Exception), e_list))
    if len(e_list) > 0:
        raise type(e_list)

def coerce_as_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

@dataclass
class _Skill(SelfDataclass):
    enabled_name: str = None
    state_name: str = None
    llm_name: str = None

    is_enabled: bool = None
    state_value: Any = None

    def pre_conditions(self):
        return [COND_ObjectExists, COND_SkillEnabled]

    def _pre_conditions(self):
        return []

    def post_conditions(self):
        return []

    def _post_conditions(self):
        return []

    def _verify_object_exists(self, sas):
        if sas.get_object_from_controller(sas.target_object) is None:
            assert sas.pre_container.get_object_by_id(sas.target_object) is not None, f"If this happened, somehow the runtime object container is desynchronized from the controller. Why????"
            return ObjectDoesNotExist(f"Object {sas.target_object} does not exist in the scene.")

    def _verify_robot_exists(self, sas):
        return True # fixme

    def _verify_skill_enabled(self):
        if not self.is_enabled:
            return SkillNotSupported(self.enabled_name)

    def verify_preconditions(self, sas):
        return verify_all_conditions(sas, self.pre_conditions() + self._pre_conditions())
        conditions = [c(sas) for c in (self.pre_conditions() + self._pre_conditions())]
        # fixme raise exceptions
        return

        exceptions = [self._verify_skill_enabled(), self._verify_robot_exists(sas), self._verify_object_exists(sas)]

        skill_specific = coerce_as_list(self._verify_precondition(sas))

        maybe_raise_many_exceptions(exceptions + skill_specific, PreconditionFailed)

    def _verify_precondition(self, sas):
        return True

    def verify_postconditions(self, sas):
        conditions = [c(sas) for c in (self.post_conditions() + self._post_conditions())]
        # fixme raise exceptions
        return

        exceptions = [self._verify_robot_exists(sas)]

        skill_specific = coerce_as_list(self._verify_postcondition(sas))

        maybe_raise_many_exceptions(exceptions + skill_specific, PostconditionFailed)

    def _verify_postcondition(self, sas):
        return True

    def _verify_isInteractable(self, sas):
        object_exists_error = self._verify_object_exists(sas)
        if object_exists_error is not None:
            return object_exists_error

        object = sas.get_object_from_controller(sas.target_object)
        if not object["isInteractable"]:
            return ObjectNotInProximity()

    def has_skill_of_name(self, skill_name: str | SimulationActionState):
        if isinstance(skill_name, SimulationActionState):
            skill_name = skill_name.skill_name

        method = getattr(self, skill_name, None)
        if callable(method):
            return True

    def get_skill_of_name(self, skill_name: str | SimulationActionState):
        if isinstance(skill_name, SimulationActionState):
            skill_name = skill_name.skill_name

        assert self.has_skill_of_name(skill_name)
        return getattr(self, skill_name)

    def output_dict(self):
        if self.state_name is not None and self.is_enabled:
            return {self.state_name: self.state_value, self.enabled_name: self.is_enabled}
        return {}

class UpdateFromAi2Thor_FLAG:
    pass

@lru_cache(maxsize=128)
def find_skillname_2_enabledname(skill_name):
    attribute_enabled_name = None
    for skill in get_all_possible_skills():
        if skill().has_skill_of_name(skill_name):
            attribute_enabled_name = skill.enabled_name
            break
    return attribute_enabled_name



@dataclass
class SkillPortfolio(SelfDataclass):
    skills: Dict[str, _Skill] = dataclasses.field(default_factory=dict)

    @classmethod
    def create(cls, skill_metadata):
        llmname_2_skillobj = get_llm_name_2_skill()

        ret = []
        for llm_name in skill_metadata:
            ret.append(llmname_2_skillobj[llm_name](is_enabled=True, state_value=False))    # fixme let the llm pick the initial state value?

        found_enablednames = [ret.enabled_name for ret in ret]


        enabledname_2_skills = get_enabledname_2_skills()
        for enabledname in enabledname_2_skills:
            if enabledname not in found_enablednames:
                ret.append(enabledname_2_skills[enabledname][0]())

        ret = {r.enabled_name: r for r in ret}

        return cls(ret)

    def find_skill(self, sas):
        attribute_enabled_name = find_skillname_2_enabledname(skill_name=sas.skill_name)
        assert attribute_enabled_name is not None
        return self.skills[attribute_enabled_name]

    def apply(self, sas: SimulationActionState) -> SimulationActionState:
        skill = self.find_skill(sas)

        sas = sas.replace(skill_object=skill)

        preconditions = skill.verify_preconditions(sas)

        # todo the stuff below probably belongs in simulator...
        if all(preconditions):
            skill_method = skill.get_skill_of_name(sas)

        result = skill_method(sas)
        if isinstance(result, _Skill):
            new_portfolio = self.effectuate_skill(result)
            target_object_instance = sas.target_object.replace(skill_portfolio=new_portfolio)
            new_object_container = sas.pre_container.update_object(target_object_instance)
        elif result is None:
            new_object_container = sas.pre_container.update_from_ai2thor(sas.get_object_list_from_controller())
        else:
            raise AssertionError("Could not recognize the result of the skill method. Make sure that the skill method returns either a skill or None.")

        sas = sas.replace(post_container=new_object_container)
        skill.verify_postconditions(sas)

        return sas


    def effectuate_skill(self, skill):
        skills = copy.deepcopy(self.skills)
        skills[skill.enabled_name] = skill
        return self.replace(skills=skills)


    def output_dict(self):
        ret = {}
        for k, v in self.skills.items():
            ret.update(v.output_dict())
        return ret

@dataclass
class _ToggleObject(_Skill):
    enabled_name: str = "toggleable"
    state_name: str = "isToggled"

    def _pre_conditions(self):
        return [COND_IsInProximity]

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

@dataclass
class PickupObject(_Ai2ThorSkill):
    # movement skill, always supported
    enabled_name: str = "pickupable"  # LLM CAN turn on/off

    llm_name: str = "can be picked up"

    def _pre_conditions(self):
        return [COND_IsInProximity, COND_AttributeEnabledInAi2Thor]

    def PickupObject(self, sas: SimulationActionState):
        sas.action_callback()
        return None

    def _post_conditions(self):
        return []# fixme make condition that checks that (1) item has isPickedUp and also that (2) robot has it in the inventory

@dataclass
class PutObject(_Ai2ThorSkill):
    enabled_name: str = "receptacle"  # LLM CAN turn on/off
    llm_name: str = "objects can be put down on this"

    def PutObject(self, sas: SimulationActionState):
        sas.action_callback()
        return None

    def _pre_conditions(self):
        return [COND_IsInProximity, COND_AttributeEnabledInAi2Thor] # fixme checks for the post conditions of PickupObject. So, the robot must have an object in its inventory.  Note: the target_object is the target surface, not the object that will be put down!

    def _post_conditions(self):
        return [] # fixme  checks that the runtime container says the object that was in the inventory is now placed on the target object "isOnTop" on whatever


@dataclass
class _OpenableObject(_Skill):
    enabled_name: str = "openable"
    state_name: str = "isOpen"

    def _pre_conditions(self):
        return [COND_IsInProximity]

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

    def _pre_conditions(self):
        return [COND_IsInProximity]

    def SliceObject(self, sas: SimulationActionState):
        return self.replace(state_value=True)

@dataclass
class BreakObject(_Skill):
    enabled_name: str = "breakable"
    state_name: str = "isBroken"

    llm_name: str = "can be broken"

    def _pre_conditions(self):
        return [COND_IsInProximity]

    def BreakObject(self, sas: SimulationActionState):
        return self.replace(state_value=True)

def _get_final_subclasses(cls):
    subclasses = cls.__subclasses__()
    if not subclasses:  # No further subclasses
        return [cls]
    final_nodes = []
    for subclass in subclasses:
        final_nodes.extend(_get_final_subclasses(subclass))  # Recursive call
    return final_nodes

def get_all_possible_skills():
    return _get_final_subclasses(_Skill)

def get_enabledname_2_skills():
    all_possible_skills = get_all_possible_skills()

    ret = {}
    for skill in all_possible_skills:
        if skill.enabled_name not in ret:
            ret[skill.enabled_name] = []
        ret[skill.enabled_name].append(skill)

    return ret

def get_enabled_2_llm_name():
    temp = get_enabledname_2_skills()
    ret = {}
    for k, v in temp.items():
        if k not in ret:
            ret[k] = []
        for _v in v:
            if _v.llm_name is None:
                continue
            ret[k].append(_v.llm_name)
        if len(ret[k]) == 0:
            del ret[k]
    return ret

def get_llm_name_2_skill():
    all_possible_skills = get_all_possible_skills()

    ret = {}
    for skill in all_possible_skills:
        if skill.llm_name is not None:
            assert skill.llm_name not in ret, f"Duplicate LLM name {skill.llm_name}"
            ret[skill.llm_name] = skill

    return ret

def get_exclusive_llm_names():
    temp = get_enabled_2_llm_name()
    return list(temp.values())


if __name__ == "__main__":
    temp = get_enabledname_2_skills()
    i=0