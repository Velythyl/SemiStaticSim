from dataclasses import dataclass
from typing import Any, Callable

from hippo.hippocontainers.runtimeobjects import RuntimeObjectContainer, RuntimeObject
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
    post_container: RuntimeObjectContainer = None

    def get_object_list_from_controller(self):
        return self.controller.last_event.metadata["objects"]

    def get_object_from_controller(self, target_object):
        # you should _verify_object_exists first
        for obj in self.get_object_list_from_controller():
            if obj["objectId"] == target_object.id:
                return obj
        return None
