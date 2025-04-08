import copy
import json
import math
import os
import threading
import time
from typing import Tuple, List

from hippo.simulation.ai2thor_metadata_reader import get_object_list_from_controller, get_robot_inventory
from hippo.simulation.runtimeobjects import RuntimeObjectContainer
from hippo.simulation.semanticverifllm.llm_semantic_verification import LLM_verify_diff, UnsafeAction, \
    LLM_verify_final_state, _LLMSemanticVerification, UnsafeFinalState
from hippo.simulation.skillsandconditions.conditions import get_slicing_implement_from_inventory, eval_conditions, \
    maybe_raise_llmcondition_exception, ConditionFailure, LLMVerificationFailure, Condition
import cv2

from hippo.simulation.skillsandconditions.sas import SimulationActionState

import re

from hippo.simulation.spatialutils.motion_planning import AStar
from hippo.utils.file_utils import get_next_file_counter
from hippo.utils.git_diff import git_diff


class Simulator:
    def __init__(self, controller, no_robots, objects: RuntimeObjectContainer, full_reachability_graph, llmverifstyle: str = "STEP", log_dir="/tmp/hipposimulation"):    # STEP or HISTORY
        self.controller = controller
        self.object_containers = [objects]

        
        self.total_exec = 0
        self.success_exec = 0
        
        self.no_robots = no_robots
        self.kill_thread = False
        
        self.action_queue = []
        self.action_listener = None

        self.task_description = None

        self.done_actions = []

        self.llmverifstyle = llmverifstyle
        self.full_reachability_graph = full_reachability_graph

        self.log_dir = log_dir


    @property
    def last_action(self):
        return self.done_actions[-1]

    @property
    def past_diffs(self):
        if len(self.object_containers) <= 1:
            return []
        ret = []
        for i, action in enumerate(self.done_actions):
            ret.append(
                git_diff(self.object_containers[i].as_llmjson(), self.object_containers[i+1].as_llmjson(), action)
            )
        return ret

    def get_diff_history(self):
        first = self.object_containers[0].as_llmjson()
        return first + "\n\n" + "\n\n".join(self.past_diffs)

    def set_task_description(self, task_description):
        self.task_description = task_description

    @property
    def current_object_container(self):
        return self.object_containers[-1]

    def pop_object_container(self):
        return self.object_containers.pop()

    def append_object_container(self, object_container):
        self.object_containers.append(object_container)

    def get_object_container_diff(self, last_action=None):
        assert len(self.object_containers) >= 2

        if last_action is None:
            last_action = self.last_action

        json_old = self.object_containers[-2].as_llmjson()
        json_new = self.object_containers[-1].as_llmjson()

        diff = git_diff(json_old, json_new, last_action)
        return diff

        return self.object_containers[-2].diff(self.object_containers[-1])

    def get_sas(self, skill_name, agent_id, target_object_id, auxiliary_object_id=None, callback=None):
        #target_object = self.current_object_container.get_object_by_id(target_object_id)

        #inventory = get_robot_inventory(self.controller, agent_id)
        #if len(inventory) == 0:
        #    auxiliary_object = None
        #else:
        #    assert len(inventory) == 1, "More than one object in the robot's inventory. Should have been caught by precondition, please report this bug."
        #if auxiliary_object_id is not None:
        #    auxiliary_object = self.current_object_container.get_object_by_id(auxiliary_object_id)
        #else:
        #    auxiliary_object = None

        skill_prettyprint = self.get_skill_prettyprint(skill_name, agent_id, target_object_id, auxiliary_object_id)
        sas = SimulationActionState(
            pre_container=self.current_object_container,
            robot=agent_id,
            target_object_id=target_object_id,
            controller=self.controller,
            action_callback=callback,
            skill_name=skill_name,
            auxiliary_object_id=auxiliary_object_id,
            skill_prettyprint=skill_prettyprint
        )

        #object_skill_portfolio = target_object.skill_portfolio
        #object_skill = object_skill_portfolio.find_skill(sas)

        #sas = sas.replace(skill_object=object_skill)
        #sas = sas.replace(skill_method=object_skill.get_skill_of_name(sas))
        #sas = sas.replace(skill_portfolio=object_skill_portfolio)

        return sas

    def preconditions_sas(self, sas):
        return eval_conditions(sas)

    def postconditions_sas(self, sas):  # todo maybe some postconditions effect change...
        raise NotImplementedError("Do we even need this? ")
        #postconditions = sas.eval_postconditions()
        #return postconditions

    def apply_sas(self, sas):
        result = sas.skill_method(sas)
        from hippo.simulation.skillsandconditions.skills_abstract import _Skill
        if isinstance(result, _Skill):
            new_portfolio = sas.skill_portfolio.effectuate_skill(result)
            target_object_instance = sas.target_object.replace(skill_portfolio=new_portfolio)
            new_object_container = sas.pre_container.update_object(target_object_instance)
        elif result is None:
            new_object_container = sas.pre_container #.update_from_ai2thor(sas.get_object_list_from_controller())
        else:
            raise AssertionError(
                "Could not recognize the result of the skill method. Make sure that the skill method returns either a skill or None.")
        new_object_container = new_object_container.update_from_ai2thor(get_object_list_from_controller(self.controller))

        sas = sas.replace(post_container=new_object_container)
        return sas

    def update_and_push_object_containers(self):
        self.append_object_container(self.current_object_container.update_from_ai2thor(get_object_list_from_controller(self.controller)))

    def get_skill_prettyprint(self, skill_name, agent_id, target_object_id, auxiliary_object_id=None):
        temp = f'{skill_name}(robot="robot{agent_id}", target_object="{target_object_id}"'

        if auxiliary_object_id is not None:
            temp += f', auxiliary_object="{auxiliary_object_id}"'
        temp += ')'
        return temp

    def apply_skill(self, skill_name, agent_id, target_object_id, auxiliary_object_id=None, callback=None):
        print(skill_name)

        sas = self.get_sas(skill_name, agent_id, target_object_id, auxiliary_object_id, callback)

        #preconditions = (
        self.preconditions_sas(sas)
        #if all(preconditions):
        sas = self.apply_sas(sas)
        #postconditions = self.postconditions_sas(sas)
        #if all(postconditions):
        self.append_object_container(sas.post_container)

        self.done_actions.append(sas.skill_prettyprint)
        return sas

    def log_llmsemantic(self, item: _LLMSemanticVerification, subpath, filepath=None):
        if filepath is None:
            filepath = get_next_file_counter(self.log_dir, subpath)
        os.makedirs(filepath, exist_ok=True)
        for k, v in vars(item).items():
            with open(f"{filepath}/{k}.txt", 'w') as f:
                f.write(f"{v}")
        with open(f"{filepath}/nametype.txt", 'w') as f:
            f.write(item.nametype)

    def log_condition(self, item: Condition, subpath, filepath=None):
        raise NotImplementedError() #if filepath is None:
        #    filepath = get_next_file_counter(self.log_dir, subpath)
        os.makedirs(filepath, exist_ok=True)
        for k, v in vars(item).items():
            with open(f"{filepath}/{k}.txt", 'w') as f:
                f.write(f"{v}")
        with open(f"{filepath}/nametype.txt", 'w') as f:
            f.write(item.nametype)
        with open(f"{filepath}/is_valid.txt", 'w') as f:
            f.write(item.success)

    def log_condition_failure(self, exception: ConditionFailure):
        path = f"{self.log_dir}/condition_failure"
        assert not os.path.exists(path)
        os.makedirs(path, exist_ok=True)    # lol

        if isinstance(exception, LLMVerificationFailure):
            item : _LLMSemanticVerification = exception.args[0]
            self.log_llmsemantic(item, None, filepath=f"{path}/llm_semantic")

            assert item.skill_prettyprint is not None

            with open(f"{path}/failure_feedback.txt", 'w') as f:
                f.write(f"Could not perform {item.skill_prettyprint} because a judge LLM said: `{item.nametype}({item.reason})`.")

        else:
            item : List[Condition] = exception.args[0]

            failure_action = item[0].sas.skill_prettyprint

            os.makedirs(f"{path}/condition_checks")
            #self.log_condition(item, subpath=f"{path}/condition")
            errors = [x.error_message() for x in item]
            with open(f"{path}/condition_checks/error_messages.txt", 'w') as f:
                f.write(json.dumps(errors))

            with open(f"{path}/failure_feedback.txt", 'w') as f:
                f.write(f"""
Could not perform {failure_action} because the following problems occurred:
```
{json.dumps(errors, indent=2)}
```
""".strip())

            with open(f"{path}/actions_so_far.txt", 'w') as f:
                f.write(json.dumps(self.done_actions))

        with open(f"{path}/exception_type.txt", 'w') as f:
            f.write(str(exception.__class__))

    def log_task_successfailure(self, llmsemantic: _LLMSemanticVerification):
        path = f"{self.log_dir}/task_success"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/is_valid.txt", 'w') as f:
            f.write(f"{llmsemantic.is_valid}")

        with open(f"{path}/maybe_feedback.txt", 'w') as f:
            if llmsemantic.is_valid:
                f.write(
                    f"A judge LLM said the plan succeeded because `{llmsemantic.nametype}({llmsemantic.reason})`.")
            else:
                f.write(
                    f"A judge LLM said the plan failed because `{llmsemantic.nametype}({llmsemantic.reason})`.")
        if not llmsemantic.is_valid:
            maybe_raise_llmcondition_exception(llmsemantic)


    def llm_verify_final_state(self):
        return
        first_state = self.object_containers[0].as_llmjson()
        last_state = self.current_object_container.as_llmjson()

        pure_diff = git_diff(first_state, last_state, "Executing_The_Plan")
        action_history = [f'{i}: {x}' for i, x in enumerate(self.done_actions)]
        action_history = "\n".join(action_history)
        diff = f"""
EXECUTED PLAN: 
{action_history}

DIFF BETWEEN FIRST AND FINAL STATES:
{pure_diff}
"""

        print("Now querying LLM to verify the safety/alignment/semantic of the final state...")
        print("The diff:")
        print(diff)
        print("Querying now...")
        llmsemantic = LLM_verify_final_state(self.task_description, diff, pure_diff, action_history)

        self.log_llmsemantic(llmsemantic, None, filepath=f"{self.log_dir}/llm_final_verif")

        if isinstance(llmsemantic, UnsafeFinalState):
            maybe_raise_llmcondition_exception(llmsemantic)

        self.log_task_successfailure(llmsemantic)

    def llm_verify_diff_alignment(self):
        return
        pure_diff = self.get_object_container_diff()
        action_history = [f'{i}: {x}' for i, x in enumerate(self.done_actions)]
        action_history = "\n".join(action_history)
        diff = f"""
ALL ACTIONS TO DATE:
{action_history}

DIFF OF LAST ACTION:
{pure_diff}
"""


        print("Now querying LLM to verify the safety/alignment/semantic of a diff...")
        print("The diff:")
        print(diff)
        print("Querying now...")
        llmsemantic = LLM_verify_diff(self.task_description, diff, pure_diff, action_history, self.done_actions[-1])    # this assumes we're synced with the done actions, which SHOULD be true. but todo verify this is true for multi agent
        self.log_llmsemantic(llmsemantic, "llm_diff_verif")
        maybe_raise_llmcondition_exception(llmsemantic)



    def _exec_actions(self):
        # create new folders to save the images from the agents
        #for i in range(self.no_robot):
        #    folder_name = "agent_" + str(i + 1)
        #    folder_path = os.path.dirname(__file__) + "/" + folder_name
        #    if not os.path.exists(folder_path):
        #        os.makedirs(folder_path)

        # create folder to store the top view images
        folder_name = "top_view"
        folder_path = os.path.dirname(__file__) + "/" + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        img_counter = 0

        while not self.kill_thread:
            if len(self.action_queue) > 0:
                try:
                    act = self.action_queue[0]
                    if act['action'] == 'ObjectNavExpertAction':
                        multi_agent_event = self.controller.step(
                            dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                        next_action = multi_agent_event.metadata['actionReturn']

                        if next_action != None:
                            multi_agent_event = self.controller.step(action=next_action, agentId=act['agent_id'], forceAction=True)
                    elif act['action'] == 'MoveAhead':
                        multi_agent_event = self.controller.step(
                            dict(action=act['action'], moveMagnitude=act['moveMagnitude'], agentId=act['agent_id']))
                        #next_action = multi_agent_event.metadata['actionReturn']

                        #if next_action != None:
                        #    multi_agent_event = self.controller.step(action=next_action, agentId=act['agent_id'], forceAction=True)
                    elif act['action'] == "GoToObject_PreConditionCheck":
                        sas = self.get_sas("GoToObject", act['agent_id'], act['objectId'], callback=None)
                        self.preconditions_sas(sas)

                    elif act['action'] == "GoToObject_PostConditionCheck":
                        sas = self.get_sas("GoToObject", act['agent_id'], act['objectId'], callback=None)
                        self.update_and_push_object_containers()
                        self.done_actions.append(sas.skill_prettyprint)
                        self.llm_verify_diff_alignment()
                        #self.postconditions_sas(sas)

                    elif act['action'] == 'MoveAhead':
                        self.controller.step(action="MoveAhead", agentId=act['agent_id'])

                    elif act['action'] == 'MoveBack':
                        self.controller.step(action="MoveBack", agentId=act['agent_id'])

                    elif act['action'] == 'RotateLeft':
                        self.controller.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])

                    elif act['action'] == 'RotateRight':
                        self.controller.step(action="RotateRight", degrees=act['degrees'],
                                                   agentId=act['agent_id'])

                    elif act["action"] == "LookUp":
                        self.controller.step(action="LookUp", agentId=act["agent_id"])

                    elif act["action"] == "LookDown":
                        self.controller.step(action="LookDown", agentId=act["agent_id"])

                    elif act['action'] == 'PickupObject':
                        def PickupObjectCallback():
                            #self.total_exec += 1
                            multi_agent_event = self.controller.step(action="PickupObject", objectId=act['objectId'],
                                                       agentId=act['agent_id'], forceAction=True)
                            if multi_agent_event.metadata['errorMessage'] != "":
                                raise Exception(multi_agent_event.metadata['errorMessage'])
                            #else:
                            #    self.success_exec += 1
                        self.apply_skill('PickupObject', agent_id=act['agent_id'], target_object_id=act['objectId'], callback=PickupObjectCallback)
                        self.llm_verify_diff_alignment()

                    elif act['action'] == 'PutObject':
                        def PutObjectCallback():
                            #self.total_exec += 1
                            multi_agent_event = self.controller.step(action="PutObject", objectId=act['objectId'],
                                                       agentId=act['agent_id'], forceAction=True)
                            if multi_agent_event.metadata['errorMessage'] != "":
                                raise Exception(multi_agent_event.metadata['errorMessage'])
                            #else:
                            #    self.success_exec += 1
                        self.apply_skill('PutObject', agent_id=act['agent_id'], target_object_id=act['objectId'], auxiliary_object_id=act["auxiliaryObjectId"], callback=PutObjectCallback)
                        self.llm_verify_diff_alignment()

                    elif act['action'] == 'ToggleObjectOn':
                        #self.total_exec += 1

                        self.apply_skill('ToggleObjectOn', agent_id=act['agent_id'], target_object_id=act['objectId'])
                        #multi_agent_event = self.controller.step(action="ToggleObjectOn", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        # todo check for return of apply_skill
                        #self.success_exec += 1
                        self.llm_verify_diff_alignment()

                    elif act['action'] == 'ToggleObjectOff':
                        self.apply_skill('ToggleObjectOff', agent_id=act['agent_id'], target_object_id=act['objectId'])
                        self.llm_verify_diff_alignment()

                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="ToggleObjectOff", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)



                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1

                    elif act['action'] == 'OpenObject':
                        self.apply_skill('OpenObject', agent_id=act['agent_id'], target_object_id=act['objectId'])
                        self.llm_verify_diff_alignment()

                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="OpenObject", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1


                    elif act['action'] == 'CloseObject':

                        self.apply_skill('CloseObject', agent_id=act['agent_id'], target_object_id=act['objectId'])
                        self.llm_verify_diff_alignment()

                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="CloseObject", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1

                    elif act['action'] == 'SliceObject':
                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="SliceObject", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1
                        #self.total_exec += 1

                        self.apply_skill('SliceObject', agent_id=act['agent_id'], target_object_id=act['objectId'])
                        knife = get_slicing_implement_from_inventory((self.controller, act["agent_id"], self.current_object_container))
                        self.apply_skill('DirtyObject', agent_id=act['agent_id'], target_object_id=knife.id)

                        actual_container = self.pop_object_container()
                        intermediary_container = self.pop_object_container()    # noqa
                        self.append_object_container(actual_container)

                        self.done_actions.pop() # removes the DirtyObject action

                        self.llm_verify_diff_alignment()

                        # multi_agent_event = self.controller.step(action="ToggleObjectOn", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        # if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        # else:
                        #
                        #self.success_exec += 1


                    elif act['action'] == 'ThrowObject':
                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="ThrowObject", moveMagnitude=7, agentId=act['agent_id'],
                        #                           forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1

                        def ThrowObjectCallback():
                            # self.total_exec += 1
                            multi_agent_event = self.controller.step(action="ThrowObject", moveMagnitude=7,
                                                                     agentId=act['agent_id'],
                                                                     forceAction=True)
                            if multi_agent_event.metadata['errorMessage'] != "":
                                raise Exception(multi_agent_event.metadata['errorMessage'])
                            # else:
                            #    self.success_exec += 1

                        self.apply_skill('ThrowObject', agent_id=act['agent_id'], target_object_id=act['objectId'],
                                         callback=ThrowObjectCallback)
                        self.llm_verify_diff_alignment()

                    elif act['action'] == 'BreakObject':
                        self.apply_skill('BreakObject', agent_id=act['agent_id'], target_object_id=act['objectId'])
                        self.llm_verify_diff_alignment()

                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="BreakObject", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1


                    elif act['action'] == 'Done':
                        self.controller.step(action="Done")
                        self.llm_verify_final_state()
                        print("Done!")
                except ConditionFailure as e:
                    print("Condition Failure! Aborting!")
                    self.controller.stop()
                    self.log_condition_failure(e)
                    raise e
                    # todo catch planning failure exceptions such as conditon failure and llm verif failure and abort plan, to provide feedback to llm


                except Exception as e:
                    raise e
                    print(e)

#                print(self.get_object_container_diff())

                for i, e in enumerate(self.controller.last_event.events):
                    cv2.imshow('agent%s' % i, e.cv2img)
                    f_name = os.path.dirname(__file__) + "/agent_" + str(i + 1) + "/img_" + str(img_counter).zfill(
                        5) + ".png"
                    cv2.imwrite(f_name, e.cv2img)
                top_view_rgb = cv2.cvtColor(self.controller.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
                cv2.imshow('Top View', top_view_rgb)
                f_name = os.path.dirname(__file__) + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
                cv2.imwrite(f_name, top_view_rgb)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                img_counter += 1
                self.action_queue.pop(0)

    def start_action_listener(self):
        self.action_listener = threading.Thread(target=self._exec_actions)
        self.action_listener.start()

    def stop_action_listener(self):
        self.kill_thread = True
        time.sleep(1)
        while self.action_listener.is_alive():
            time.sleep(0.5)
        self.action_listener.join()

    def push_action(self, action):
        self.action_queue.append(action)

    def _get_robot_name(self, robot):
        return robot['name']

    def _get_robot_id(self, robot):
        return int(self._get_robot_name(robot)[-1]) - 1

    def _get_object_id(self, target_obj):
        objs = list(set([obj["objectId"] for obj in self.controller.last_event.metadata["objects"]]))

        sw_obj_id = target_obj

        for obj in objs:
            match = re.match(target_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break  # find the first instance

        return sw_obj_id

    def _get_object_center(self, target_obj):
        objs_centers = {obj["objectId"]: obj["axisAlignedBoundingBox"]["center"] for obj in self.controller.last_event.metadata["objects"]}
        # list([obj["axisAlignedBoundingBox"]["center"] for obj in self.controller.last_event.metadata["objects"]])
        return objs_centers.get(target_obj, None)

    # ========= SKILLS =========

    def GoToObject(self, robots, dest_obj):
        # todo https://chat.deepseek.com/a/chat/s/411a780c-246e-4909-ac93-48ad5f66e14f
        print("Going to ", dest_obj)
        # check if robots is a list

        if not isinstance(robots, list):
            # convert robot to a list
            robots = [robots]


        dest_obj_id = self._get_object_id(dest_obj)

        for ia, robot in enumerate(robots):
            self.push_action(
                {
                    'action': 'GoToObject_PreConditionCheck',
                    'agent_id': self._get_robot_id(robot),
                    'objectId': dest_obj_id
                }
            )

        time.sleep(0.5)

        dest_obj_center = self._get_object_center(dest_obj_id)
        dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]

        dest_obj_aabb = {obj["objectId"]: obj["axisAlignedBoundingBox"] for obj in
                        self.controller.last_event.metadata["objects"]}.get(dest_obj_id, None)['size']
        dest_obj_aabb = (dest_obj_aabb['x'], dest_obj_aabb['y'], dest_obj_aabb['z'])

        def get_cur_pos(robot):
            metadata = self.controller.last_event.events[self._get_robot_id(robot)].metadata
            pos = [metadata["agent"]["position"]["x"],
                metadata["agent"]["position"]["y"],
                metadata["agent"]["position"]["z"]]

            return pos

        def dist_to_goal(robot):
            # Robot position and half-size
            robot_center = get_cur_pos(robot)
            robot_center = (robot_center[0], robot_center[2])
            robot_center = np.array(robot_center)
            robot_half_size = np.array([0.3, 0.3])  # Replace with actual half-size

            # Object position and half-size
            obj_center = (dest_obj_pos[0], dest_obj_pos[2])
            obj_center = np.array(obj_center)
            obj_half_size = (dest_obj_aabb[0], dest_obj_aabb[2])
            obj_half_size = np.array(obj_half_size) / 2  # Replace with actual half-size

            # Calculate the distance between centers
            delta = robot_center - obj_center

            # Calculate the overlap in each dimension
            overlap_x = abs(delta[0]) - (robot_half_size[0] + obj_half_size[0])
            overlap_y = abs(delta[1]) - (robot_half_size[1] + obj_half_size[1])

            # If rectangles overlap, distance is negative (you might want to handle this case differently)
            if overlap_x < 0 and overlap_y < 0:
                return max(overlap_x, overlap_y)  # negative value indicates penetration depth
            elif overlap_x < 0:
                return overlap_y
            elif overlap_y < 0:
                return overlap_x
            else:
                return np.sqrt(overlap_x ** 2 + overlap_y ** 2)

        def dist_robot_2_node(robot, node):
            return np.linalg.norm(np.array(get_cur_pos(robot)) - np.array(node))

        def get_robot_location_dict(robot):
            metadata = self.controller.last_event.events[self._get_robot_id(robot)].metadata
            robot_location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            return robot_location

        def rotate_to_face_node(robot, node):
            # align the robot once goal is reached
            # compute angle between robot heading and object
            robot_location = get_robot_location_dict(robot)

            robot_object_vec = [node[0] - robot_location['x'],
                                node[2] - robot_location['z']]
            y_axis = [0, 1]
            unit_y = y_axis / np.linalg.norm(y_axis)
            unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)

            angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
            angle = 360 * angle / (2 * np.pi)
            angle = (angle + 360) % 360
            rot_angle = angle - robot_location['rotation']

            if rot_angle > 0:
                self.push_action({'action': 'RotateRight', 'degrees': abs(rot_angle),
                                  'agent_id': self._get_robot_id(robot)})
            else:
                self.push_action({'action': 'RotateLeft', 'degrees': abs(rot_angle),
                                  'agent_id': self._get_robot_id(robot)})

        astars = [AStar(dest_obj_pos, self.full_reachability_graph,
                                                self.current_object_container) for robot in robots]

        goal_thresh = 0.4
        def get_generators():
            new_generators = []
            for ia, robot in enumerate(robots):
                if dist_to_goal(robot) > goal_thresh:
                    from hippo.simulation.spatialutils.motion_planning import astar
                    new_generators.append(astars[ia].generate_path(get_cur_pos(robot)))
                else:
                    new_generators.append(None)
            return new_generators
        generators = get_generators()

        max_num_tries = 10
        num_tries = 0
        while True:
            if all([x is None for x in generators]):
                break
            try:
                for ia, robot in enumerate(robots):
                    if generators[ia] is not None:
                        node = next(generators[ia])

                        rotate_to_face_node(robot, node)
                        self.push_action(
                            {'action': 'MoveAhead', 'moveMagnitude': dist_robot_2_node(robot, node),
                             'agent_id': self._get_robot_id(robot)})

            except StopIteration:
                generators = get_generators()
                num_tries += 1
                if num_tries > max_num_tries:
                    raise AssertionError("Motion planning failure, fix the astar path planner")

            time.sleep(0.5)

            ALL_DONE = True
            for ia, robot in enumerate(robots):
                d = dist_to_goal(robot)
                print(f"Going to {dest_obj_id}, distance:", d)
                if not d < goal_thresh:
                    ALL_DONE = False
            if ALL_DONE:
                break

        for ia, robot in enumerate(robots):
            rotate_to_face_node(robot, dest_obj_pos)

        def LookUpDownAtObject(robot, agent_id):
            # todo make this its own function and call it after every object interaction...
            robot_location = get_robot_location_dict(robot)
            dy = dest_obj_pos[1] - robot_location["y"]
            # Compute yaw rotation
            dx = dest_obj_pos[0] - robot_location["x"]
            dz = dest_obj_pos[2] - robot_location["z"]

            horizontal_dist = math.sqrt(dx ** 2 + dz ** 2)
            pitch = math.degrees(math.atan2(dy, horizontal_dist))

            # Adjust camera pitch
            current_horizon = robot_location["horizon"]
            if pitch > current_horizon:
                self.push_action({"action": "LookUp", "agent_id": agent_id})
            else:
                self.push_action({"action": "LookDown", "agent_id": agent_id})

        def get_dest_obj(agent_id):
            for obj in self.controller.last_event.events[agent_id].metadata["objects"]:
                if obj["objectId"] == dest_obj_id:
                    return obj
            raise AssertionError("Could not find destination object?!")

        for ia, robot in enumerate(robots):
            NUM_TRIES = 0
            MAX_NUM_TRIES = 10
            while not get_dest_obj(self._get_robot_id(robot))["visible"]:
                LookUpDownAtObject(robot, self._get_robot_id(robot))
                time.sleep(0.5)
                if NUM_TRIES > MAX_NUM_TRIES:
                    break

        for ia, robot in enumerate(robots):
            self.push_action(
                {
                    'action': 'GoToObject_PostConditionCheck',
                    'agent_id': self._get_robot_id(robot),
                    'objectId': dest_obj_id
                }
            )
        print("Reached: ", dest_obj)

    def PickupObject(self, robot, pick_obj):
        self.push_action({'action': 'PickupObject', 'objectId': self._get_object_id(pick_obj), 'agent_id': self._get_robot_id(robot)})

    def PutObject(self, robot, put_obj, recp):
        return self.push_action(
            {'action': 'PutObject', 'objectId': self._get_object_id(recp), 'agent_id': self._get_robot_id(robot),
             'auxiliaryObjectId': self._get_object_id(put_obj)})

    def SwitchOn(self, robot, sw_obj):
        self.push_action({'action': 'ToggleObjectOn', 'objectId': self._get_object_id(sw_obj), 'agent_id': self._get_robot_id(robot)})

    def SwitchOff(self, robot, sw_obj):
        self.push_action({'action': 'ToggleObjectOff', 'objectId': self._get_object_id(sw_obj), 'agent_id': self._get_robot_id(robot)})

    def OpenObject(self, robot, sw_obj):
        self.push_action({'action': 'OpenObject', 'objectId': self._get_object_id(sw_obj), 'agent_id': self._get_robot_id(robot)})

    def CloseObject(self, robot, sw_obj):
        self.push_action({'action': 'CloseObject', 'objectId': self._get_object_id(sw_obj), 'agent_id': self._get_robot_id(robot)})

    def BreakObject(self, robot, sw_obj):
        self.push_action({'action': 'BreakObject', 'objectId': self._get_object_id(sw_obj), 'agent_id': self._get_robot_id(robot)})

    def SliceObject(self, robot, sw_obj):
        self.push_action({'action': 'SliceObject', 'objectId': self._get_object_id(sw_obj), 'agent_id': self._get_robot_id(robot)})

    def CleanObject(self, robot, sw_obj):
        self.push_action({'action': 'CleanObject', 'objectId': self._get_object_id(sw_obj), 'agent_id': self._get_robot_id(robot)})

    def ThrowObject(self, robot, sw_obj):
        self.push_action({'action': 'ThrowObject', 'objectId': self._get_object_id(sw_obj), 'agent_id': self._get_robot_id(robot)})
        time.sleep(1)

    def Done(self):
        self.push_action({'action': 'Done'})
        time.sleep(1)

from scipy.spatial import distance
import numpy as np
def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[i] #(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5