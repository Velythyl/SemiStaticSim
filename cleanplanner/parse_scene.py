import dataclasses
from dataclasses import field
from typing import Union, Any, List, Dict

from cleanplanner.resources.robots import robots
from hippo.ai2thor_hippo_controller import resolve_scene_id
from hippo.utils.selfdataclass import SelfDataclass


@dataclasses.dataclass
class SceneTask(SelfDataclass):
    scene_id: str
    scene: Union[str, dict]

    tasks: List[str]
    robots_ids: List[int]
    robots: List[Any]

    @property
    def sc_ta_ro_ID(self):
        ta = '-'.join(self.tasks)
        ro = '-'.join(map(str, self.robots_ids))
        return f"SC-{self.scene_id}_TA-{ta}_RO-{ro}"

    plans: List[Any] = field(default_factory=list)
    feedbacks: List[Any] = field(default_factory=list)
    successes: List[bool] = field(default_factory=list)

def parse_floorplan(scene_id, tasks, scene_robots):
    assert isinstance(scene_id, str)

    scene = resolve_scene_id(scene_id)

    available_robots = []
    for i, r_id in enumerate(scene_robots):
        rob = robots[r_id - 1]
        # rename the robot
        rob['name'] = 'robot' + str(i + 1)
        available_robots.append(rob)

    return SceneTask(scene_id, scene, list(tasks), list(scene_robots), available_robots)

@dataclasses.dataclass
class PlanLog(SelfDataclass):
    llm: str
    scenetask: SceneTask
    num_input_tokens: int
    num_output_tokens: int
    code_plan: str
    llm_outputs: Dict[str, str]
