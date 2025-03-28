import os
import threading
import time

from hippo.simulation.ai2thor_metadata_reader import get_object_list_from_controller, get_robot_inventory
from hippo.simulation.runtimeobjects import RuntimeObjectContainer

import cv2

from hippo.simulation.skillsandconditions.sas import SimulationActionState

import re
class Simulator:
    def __init__(self, controller, no_robots, objects: RuntimeObjectContainer):
        self.controller = controller
        self.transations = [objects]
        
        self.total_exec = 0
        self.success_exec = 0
        
        self.no_robots = no_robots
        self.kill_thread = False
        
        self.action_queue = []
        self.action_listener = None

    @property
    def objects(self):
        return self.transations[-1]

    def get_sas(self, skill_name, agent_id, target_object_id, auxiliary_object_id=None, callback=None):
        target_object = self.objects.get_object_by_id(target_object_id)

        #inventory = get_robot_inventory(self.controller, agent_id)
        #if len(inventory) == 0:
        #    auxiliary_object = None
        #else:
        #    assert len(inventory) == 1, "More than one object in the robot's inventory. Should have been caught by precondition, please report this bug."
        if auxiliary_object_id is not None:
            auxiliary_object = self.objects.get_object_by_id(auxiliary_object_id)
        else:
            auxiliary_object = None

        sas = SimulationActionState(
            pre_container=self.objects,
            robot=agent_id,
            target_object=target_object,
            controller=self.controller,
            action_callback=callback,
            skill_name=skill_name,
            auxiliary_object=auxiliary_object
        )

        object_skill_portfolio = target_object.skill_portfolio
        object_skill = object_skill_portfolio.find_skill(sas)

        sas = sas.replace(skill_object=object_skill)
        sas = sas.replace(skill_method=object_skill.get_skill_of_name(sas))
        sas = sas.replace(skill_portfolio=object_skill_portfolio)

        return sas

    def preconditions_sas(self, sas):
        preconditions = sas.eval_preconditions()
        return preconditions

    def postconditions_sas(self, sas):  # todo maybe some postconditions effect change...
        postconditions = sas.eval_postconditions()
        return postconditions

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


    def apply_skill(self, skill_name, agent_id, target_object_id, auxiliary_object_id=None, callback=None):
        sas = self.get_sas(skill_name, agent_id, target_object_id, auxiliary_object_id, callback)

        preconditions = self.preconditions_sas(sas)
        if all(preconditions):
            sas = self.apply_sas(sas)
            postconditions = self.postconditions_sas(sas)
            if all(postconditions):
                self.transations.append(sas.post_container)

                print(self.transations[-2].diff(self.transations[-1]))
        return


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
                    elif act['action'] == "GoToObject_PreConditionCheck":
                        sas = self.get_sas("GoToObject", act['agent_id'], act['objectId'], callback=None)
                        self.preconditions_sas(sas)

                    elif act['action'] == "GoToObject_PostConditionCheck":
                        sas = self.get_sas("GoToObject", act['agent_id'], act['objectId'], callback=None)
                        self.postconditions_sas(sas)

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
                            self.total_exec += 1
                            multi_agent_event = self.controller.step(action="PickupObject", objectId=act['objectId'],
                                                       agentId=act['agent_id'], forceAction=True)
                            if multi_agent_event.metadata['errorMessage'] != "":
                                print(multi_agent_event.metadata['errorMessage'])
                            else:
                                self.success_exec += 1
                        self.apply_skill('PickupObject', agent_id=act['agent_id'], target_object_id=act['objectId'], callback=PickupObjectCallback)

                    elif act['action'] == 'PutObject':
                        def PutObjectCallback():
                            self.total_exec += 1
                            multi_agent_event = self.controller.step(action="PutObject", objectId=act['objectId'],
                                                       agentId=act['agent_id'], forceAction=True)
                            if multi_agent_event.metadata['errorMessage'] != "":
                                print(multi_agent_event.metadata['errorMessage'])
                            else:
                                self.success_exec += 1
                        self.apply_skill('PutObject', agent_id=act['agent_id'], target_object_id=act['objectId'], auxiliary_object_id=act["auxiliaryObjectId"], callback=PutObjectCallback)

                    elif act['action'] == 'ToggleObjectOn':
                        self.total_exec += 1

                        self.apply_skill('ToggleObjectOn', agent_id=act['agent_id'], target_object_id=act['objectId'])
                        #multi_agent_event = self.controller.step(action="ToggleObjectOn", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        # todo check for return of apply_skill
                        self.success_exec += 1

                    elif act['action'] == 'ToggleObjectOff':
                        self.total_exec += 1
                        multi_agent_event = self.controller.step(action="ToggleObjectOff", objectId=act['objectId'],
                                                   agentId=act['agent_id'], forceAction=True)



                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(multi_agent_event.metadata['errorMessage'])
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'OpenObject':
                        self.total_exec += 1
                        multi_agent_event = self.controller.step(action="OpenObject", objectId=act['objectId'],
                                                   agentId=act['agent_id'], forceAction=True)
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(multi_agent_event.metadata['errorMessage'])
                        else:
                            self.success_exec += 1


                    elif act['action'] == 'CloseObject':
                        self.total_exec += 1
                        multi_agent_event = self.controller.step(action="CloseObject", objectId=act['objectId'],
                                                   agentId=act['agent_id'], forceAction=True)
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(multi_agent_event.metadata['errorMessage'])
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'SliceObject':
                        #self.total_exec += 1
                        #multi_agent_event = self.controller.step(action="SliceObject", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        #if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        #else:
                        #    self.success_exec += 1
                        self.total_exec += 1

                        self.apply_skill('SliceObject', agent_id=act['agent_id'], target_object_id=act['objectId'])
                        # multi_agent_event = self.controller.step(action="ToggleObjectOn", objectId=act['objectId'],
                        #                           agentId=act['agent_id'], forceAction=True)
                        # if multi_agent_event.metadata['errorMessage'] != "":
                        #    print(multi_agent_event.metadata['errorMessage'])
                        # else:
                        # todo check for return of apply_skill
                        self.success_exec += 1


                    elif act['action'] == 'ThrowObject':
                        self.total_exec += 1
                        multi_agent_event = self.controller.step(action="ThrowObject", moveMagnitude=7, agentId=act['agent_id'],
                                                   forceAction=True)
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(multi_agent_event.metadata['errorMessage'])
                        else:
                            self.success_exec += 1

                    elif act['action'] == 'BreakObject':
                        self.total_exec += 1
                        multi_agent_event = self.controller.step(action="BreakObject", objectId=act['objectId'],
                                                   agentId=act['agent_id'], forceAction=True)
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(multi_agent_event.metadata['errorMessage'])
                        else:
                            self.success_exec += 1


                    elif act['action'] == 'Done':
                        self.controller.step(action="Done")


                except Exception as e:
                    raise e
                    print(e)

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

    def _get_agent_id(self, robot):
        return int(robot['name'][-1]) - 1

    def _get_position(self, robot):
        return robot['position']

    def _get_rotation(self, robot):
        return robot['rotation']

    def _get_object_id(self, target_obj):
        objs = list(set([obj["objectId"] for obj in self.controller.last_event.metadata["objects"]]))

        sw_obj_id = target_obj

        for obj in objs:
            match = re.match(target_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break  # find the first instance

        return sw_obj_id


    # ========= SKILLS =========

    def SwitchOn(self, robot, object):
        self.push_action({'action': 'ToggleObjectOn', 'objectId': self._get_object_id(object), 'agent_id': self._get_agent_id(robot)})


