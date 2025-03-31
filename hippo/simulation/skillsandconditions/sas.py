from dataclasses import dataclass
from typing import Any, Callable

from hippo.simulation.ai2thor_metadata_reader import get_object_list_from_controller
from hippo.simulation.runtimeobjects import RuntimeObjectContainer, RuntimeObject
from hippo.utils.selfdataclass import SelfDataclass



@dataclass
class SimulationActionState(SelfDataclass):
    pre_container: RuntimeObjectContainer
    skill_name: str
    robot: str
    target_object: RuntimeObject
    controller: Any
    action_callback: Callable = None
    skill_object: Any = None
    skill_method: Callable = None
    post_container: RuntimeObjectContainer = None
    skill_portfolio: Any = None
    auxiliary_object: RuntimeObject = None

    def get_object_list_from_controller(self):
        return self.controller.last_event.metadata["objects"]

    def get_object_from_controller(self, target_object):
        # you should _verify_object_exists first
        for obj in get_object_list_from_controller(self.controller):
            if obj["objectId"] == target_object.id:
                return obj
        return None

    #def eval_preconditions(self):
    #    preconditions = [c(self) for c in self.skill_object.pre_conditions]
    #    return self.maybe_raise_condition_exception(preconditions, PreconditionFailure, MultiplePreconditionFailure)

    #def eval_postconditions(self):
    #    postconditions = [c(self) for c in self.skill_object.post_conditions]
    #    return self.maybe_raise_condition_exception(postconditions, PostconditionFailure, MultiplePostconditionFailure)

    """
    def maybe_raise_condition_exception(self, condlist, single_exception_class, multiple_exception_class):
        if all(condlist):
            return condlist

        errors = []
        for cond in condlist:
            if cond.success is False:
                errors.append(cond.error_message())

        if len(errors) == 1:
            raise single_exception_class(errors[0])
        else:
            raise multiple_exception_class(errors)"""
