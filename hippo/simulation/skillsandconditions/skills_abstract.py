import copy
import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

from hippo.simulation.skillsandconditions.conditions import COND_ObjectExists, COND_SkillEnabled, verify_all_conditions
from hippo.simulation.skillsandconditions.sas import SimulationActionState
from hippo.utils.selfdataclass import SelfDataclass


@dataclass
class _Skill(SelfDataclass):
    enabled_name: str = None
    state_name: str = None
    llm_name: str = None

    is_enabled: bool = None
    state_value: Any = None

    def __post_init__(self):
        import_concrete_skills()

    @property
    def pre_conditions(self):
        return []

    @property
    def post_conditions(self):
        return []

    def verify_preconditions(self, sas):
        return verify_all_conditions(sas, self.pre_conditions)

    def verify_postconditions(self, sas):
        return verify_all_conditions(sas, self.post_conditions)

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


@lru_cache(maxsize=128)
def find_skillname_2_enabledname(skill_name):
    attribute_enabled_name = None
    for skill in get_all_possible_skills():
        if skill().has_skill_of_name(skill_name):
            attribute_enabled_name = skill.enabled_name
            break
    return attribute_enabled_name

def import_concrete_skills():
    import hippo.simulation.skillsandconditions.skills  # noqa
    return


@dataclass
class SkillPortfolio(SelfDataclass):
    skills: Dict[str, _Skill] = dataclasses.field(default_factory=dict)

    @classmethod
    def create(cls, skill_metadata):
        import_concrete_skills()

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
        import hippo.simulation.skillsandconditions.skills  # noqa
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
