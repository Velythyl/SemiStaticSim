import copy
import json
import math
import os
import sys
import threading
import time
from typing import Tuple, List, Dict

from hippo.simulation.ai2thor_metadata_reader import get_object_list_from_controller, get_robot_inventory, \
    get_object_position_from_controller, get_robot_position_from_controller, get_object_size_from_controller, \
    get_object_aabb_from_controller, get_object_from_controller
from hippo.simulation.runtimeobjects import RuntimeObjectContainer
from hippo.simulation.semanticverifllm.llm_semantic_verification import LLM_verify_diff, UnsafeAction, \
    LLM_verify_final_state, _LLMSemanticVerification, UnsafeFinalState
from hippo.simulation.singlefilelog import log_scenedict_to_file
from hippo.simulation.skillsandconditions.conditions import get_slicing_implement_from_inventory, eval_conditions, \
    maybe_raise_llmcondition_exception, ConditionFailure, LLMVerificationFailure, Condition
import cv2

from hippo.simulation.skillsandconditions.sas import SimulationActionState

import re

from hippo.simulation.spatialutils.motion_planning import AStar
from hippo.utils.file_utils import get_next_file_counter
from hippo.utils.git_diff import git_diff


class Simulator:
    def __init__(self, controller, no_robots, objects: RuntimeObjectContainer, full_reachability_graph, llmverifstyle: str = "STEP"):    # STEP or HISTORY
        self.controller = controller
        self.object_containers = [objects]

        log_scenedict_to_file(0, objects.as_llmjson())

        
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

        self.roblock = {}

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


    def llm_verify_final_state(self):
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

        print(llmsemantic.response)

        maybe_raise_llmcondition_exception(llmsemantic)

    def llm_verify_diff_alignment(self):
        log_scenedict_to_file(len(self.object_containers)-1, self.current_object_container.as_llmjson())
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
                        print("moved ahead, releasing robot..")
                        self._release_robot(act['agent_id'])
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

                    #elif act['action'] == 'MoveAhead':
                    #    self.controller.step(action="MoveAhead", agentId=act['agent_id'])

                    #elif act['action'] == 'MoveBack':
                    #    self.controller.step(action="MoveBack", agentId=act['agent_id'])

                    elif act['action'] == 'RotateLeft':
                        self.controller.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])
                        self._release_robot(robot=act['agent_id'])

                    elif act['action'] == 'RotateRight':
                        self.controller.step(action="RotateRight", degrees=act['degrees'],
                                                   agentId=act['agent_id'])
                        self._release_robot(robot=act['agent_id'])

                    elif act["action"] == "LookUp":
                        self.controller.step(action="LookUp", agentId=act["agent_id"])
                        self._release_robot(act['agent_id'])

                    elif act["action"] == "LookDown":
                        self.controller.step(action="LookDown", agentId=act["agent_id"])
                        self._release_robot(act['agent_id'])

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
                        def ToggleObjectOn():
                            multi_agent_event = self.controller.step(action="ToggleObjectOn", objectId=act['objectId'],
                                                                    agentId=act['agent_id'], forceAction=True)

                        self.apply_skill('ToggleObjectOn', agent_id=act['agent_id'], target_object_id=act['objectId'], callback=ToggleObjectOn)
                        #multi_agent_event = self.controller.step(action="ToggleObjectOn", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        # todo check for return of apply_skill
                        #self.success_exec += 1
                        self.llm_verify_diff_alignment()

                    elif act['action'] == 'ToggleObjectOff':
                        def ToggleObjectOff():
                            multi_agent_event = self.controller.step(action="ToggleObjectOff", objectId=act['objectId'],
                                                                                                agentId=act['agent_id'], forceAction=True)

                        self.apply_skill('ToggleObjectOff', agent_id=act['agent_id'], target_object_id=act['objectId'], callback=ToggleObjectOff)
                        self.llm_verify_diff_alignment()

                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="ToggleObjectOff", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)



                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1

                    elif act['action'] == 'OpenObject':
                        def OpenObject():
                            multi_agent_event = self.controller.step(action="OpenObject", objectId=act['objectId'],
                                                                                               agentId=act['agent_id'], forceAction=True)

                        x = get_object_aabb_from_controller(self.controller, act['objectId'])
                        self.apply_skill('OpenObject', agent_id=act['agent_id'], target_object_id=act['objectId'], callback=OpenObject)
                        y = get_object_aabb_from_controller(self.controller, act['objectId'])
                        self.llm_verify_diff_alignment()
                        self._release_robot(act['agent_id'])

                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="OpenObject", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1


                    elif act['action'] == 'CloseObject':
                        def CloseObject():
                            multi_agent_event = self.controller.step(action="CloseObject", objectId=act['objectId'],
                                                                                                agentId=act['agent_id'], forceAction=True)

                        self.apply_skill('CloseObject', agent_id=act['agent_id'], target_object_id=act['objectId'], callback=CloseObject)
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

                        def SliceObject():
                            self.controller.step(action="SliceObject", objectId=act['objectId'],
                                                                            agentId=act['agent_id'], forceAction=True)


                        self.apply_skill('SliceObject', agent_id=act['agent_id'], target_object_id=act['objectId'], callback=SliceObject)
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
                        def BreakObject():
                            multi_agent_event = self.controller.step(action="BreakObject", objectId=act['objectId'],
                                                                                                agentId=act['agent_id'], forceAction=True)

                        self.apply_skill('BreakObject', agent_id=act['agent_id'], target_object_id=act['objectId'], callback=BreakObject)
                        self.llm_verify_diff_alignment()

                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="BreakObject", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1

                    elif act['action'] == "AbortPlan":
                        temp = IncorrectTaskDescription(
                            task_description = self.task_description,
                            diff = "",
                            response ="",
                            reason=act['reason']
                        )
                        maybe_raise_llmcondition_exception(temp)


                    elif act['action'] == 'Done':
                        self.controller.step(action="Done")
                        self.llm_verify_final_state()
                        print("Done!")
                        raise Exception("Done!")
                except ConditionFailure as e:
                    print("Condition Failure! Aborting!")
                    self.controller.stop()
                    os._exit(0)
                    raise e

                except Exception as e:
                    os._exit(0)
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

    def _lock_robot(self, robot):
        if not isinstance(robot, int):
            robot = self._get_robot_id(robot)
        assert isinstance(robot, int)

        if robot not in self.roblock:
            self.roblock[robot] = threading.Lock()

        self.roblock[robot].acquire()

    def _await_robot(self, robot):
        self._lock_robot(robot)
        self._release_robot(robot)

    def _notify_robot(self, robot):
        self._release_robot(robot)

    def _release_robot(self, robot):
        if not isinstance(robot, int):
            robot = self._get_robot_id(robot)
        assert isinstance(robot, int)

        try:
            self.roblock[robot].release()
        except RuntimeError as e:
            pass

    def _get_object_id(self, target_obj):
        objs = list(set([obj["objectId"] for obj in self.controller.last_event.metadata["objects"]]))

        for obj in objs:
            if obj == target_obj:
                return target_obj

        for obj in objs:
            if target_obj.lower() in obj.lower():
                return obj

        sw_obj_id = target_obj

        for obj in objs:
            match = re.match(target_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break  # find the first instance

        return sw_obj_id



    def _get_robot_location_dict(self,  robot):
        metadata = self.controller.last_event.events[self._get_robot_id(robot)].metadata
        robot_location = {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"]}
        return robot_location


    def _get_object_aabb(self, object_id):
        return {obj["objectId"]: obj["axisAlignedBoundingBox"] for obj in
         self.controller.last_event.metadata["objects"]}.get(object_id, {})

    # ========= SKILLS =========

    def AbortPlan(self, robot, reason):
        self.push_action({
            "action": "AbortPlan",
            "reason": reason,
        })

    def GoToObject(self, robot, dest_obj):
        # todo https://chat.deepseek.com/a/chat/s/411a780c-246e-4909-ac93-48ad5f66e14f
        print("Going to ", dest_obj)
        # check if robots is a list

        dest_obj_id = self._get_object_id(dest_obj)
        self.push_action(
            {
                'action': 'GoToObject_PreConditionCheck',
                'agent_id': self._get_robot_id(robot),
                'objectId': dest_obj_id
            }
        )

        dest_obj_pos = get_object_position_from_controller(self.controller, dest_obj_id)
        dest_obj_size = get_object_size_from_controller(self.controller, dest_obj_id)

        def dist_to_goal(obj_pos, obj_size):
            from hippo.simulation.spatialutils.proximity_spatial_funcs import bbox_dist
            obj_pos = np.array(obj_pos)
            obj_size = np.array(obj_size)
            return bbox_dist(
                np.array(get_robot_position_from_controller(self.controller, self._get_robot_id(robot))), np.array([0.6, 0.95, 0.6]),
                obj_pos, obj_size
            )

        def RotateToNode(robot, node):
            # align the robot once goal is reached
            # compute angle between robot heading and object
            robot_location = self._get_robot_location_dict(robot)

            robot_object_vec = [node[0] - robot_location['x'],
                                node[2] - robot_location['z']]
            y_axis = [0, 1]
            unit_y = y_axis / np.linalg.norm(y_axis)
            unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)

            angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
            angle = 360 * angle / (2 * np.pi)
            angle = (angle + 360) % 360
            rot_angle = angle - robot_location['rotation']

            self._lock_robot(robot)
            if rot_angle > 0:
                self.push_action({'action': 'RotateRight', 'degrees': abs(rot_angle),
                                  'agent_id': self._get_robot_id(robot)})
            else:
                self.push_action({'action': 'RotateLeft', 'degrees': abs(rot_angle),
                                  'agent_id': self._get_robot_id(robot)})
            self._await_robot(robot)

        goal_thresh = 0.75

        def is_dest_obj_visible():
            return get_object_from_controller(self.controller, dest_obj_id)["visible"]

        def are_we_done():
            d = dist_to_goal(dest_obj_pos, dest_obj_size)
            obj_insideness = None # self.current_object_container.get_obj2id_that_obj1id_is_inside_of(dest_obj_id) todo
            if obj_insideness is not None:
                d2 = dist_to_goal(get_object_position_from_controller(self.controller, obj_insideness), get_object_size_from_controller(self.controller, obj_insideness))
            else:
                d2 = np.inf
            return (
                    d<goal_thresh or
                    (dest_obj_id in get_robot_inventory(self.controller, self._get_robot_id(robot))) or
                    d2 < goal_thresh
            )

        if are_we_done():
            print(f"Was going to {dest_obj_id}, but already next to object. Will only adjust camera.")
        else:
            from hippo.simulation.spatialutils.motion_planning import astar
            path_gen = astar(get_robot_position_from_controller(self.controller, self._get_robot_id(robot)), dest_obj_pos, self.full_reachability_graph,
                                            self.current_object_container)
            NUM_RETRIES = 0
            MAX_NUM_RETRIES = 3
            NUM_OUTPUT_NODES = 0
            MAX_OUTPUT_NODES = 3
            #OLD_NODE = None
            while True:
                try:
                    node = next(path_gen)

                    print(node)

                    moveMagnitude =  np.linalg.norm(
                        np.array(get_robot_position_from_controller(self.controller, self._get_robot_id(robot)))
                        - np.array(node)
                    )
                    RotateToNode(robot, node)
                    self._lock_robot(robot)
                    self.push_action(
                        {'action': 'MoveAhead', 'moveMagnitude': moveMagnitude,
                         'agent_id': self._get_robot_id(robot)})
                    self._await_robot(robot)

                    d = dist_to_goal(dest_obj_pos, dest_obj_size)
                    print(f"Going to {dest_obj_id}, distance:", d)

                    #if OLD_NODE is not None:
                    #    if get_robot_position_from_controller(self.controller, self._get_robot_id(robot)):

                    #OLD_NODE = node

                    NUM_OUTPUT_NODES += 1
                    if NUM_OUTPUT_NODES >= MAX_OUTPUT_NODES:
                        path_gen = astar(get_robot_position_from_controller(self.controller, self._get_robot_id(robot)),
                                         dest_obj_pos, self.full_reachability_graph,
                                         self.current_object_container)
                        NUM_OUTPUT_NODES = 0

                except StopIteration:
                    if not are_we_done():
                        if NUM_RETRIES < MAX_NUM_RETRIES:
                            path_gen = astar(get_robot_position_from_controller(self.controller, self._get_robot_id(robot)), dest_obj_pos, self.full_reachability_graph,
                                                self.current_object_container)
                            NUM_RETRIES += 1
                        else:
                            assert are_we_done(), "Possible motion planning failure, fix the astar path planner"
                    else:
                        break

        RotateToNode(robot, dest_obj_pos)

        def LookUpDownAtObject(robot, agent_id):
            # todo make this its own function and call it after every object interaction...
            robot_location = self._get_robot_location_dict(robot)
            dy = dest_obj_pos[1] - robot_location["y"]
            # Compute yaw rotation
            dx = dest_obj_pos[0] - robot_location["x"]
            dz = dest_obj_pos[2] - robot_location["z"]

            horizontal_dist = math.sqrt(dx ** 2 + dz ** 2)
            pitch = math.degrees(math.atan2(dy, horizontal_dist))

            # Adjust camera pitch
            current_horizon = robot_location["horizon"]

            self._lock_robot(robot)
            if pitch > current_horizon:
                self.push_action({"action": "LookUp", "agent_id": agent_id})
            else:
                self.push_action({"action": "LookDown", "agent_id": agent_id})
            self._await_robot(robot)



        NUM_TRIES = 0
        MAX_NUM_TRIES = 10
        while not is_dest_obj_visible():
            LookUpDownAtObject(robot, self._get_robot_id(robot))
            NUM_TRIES += 1
            if NUM_TRIES > MAX_NUM_TRIES:
                break

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
        self._lock_robot(robot)
        self.push_action({'action': 'OpenObject', 'objectId': self._get_object_id(sw_obj), 'agent_id': self._get_robot_id(robot)})
        self._await_robot(robot)

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