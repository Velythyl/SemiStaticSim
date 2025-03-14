import os
import threading
import time

from hippo.hippocontainers.runtimeobjects import RuntimeObjectContainer

import cv2

from hippo.hippocontainers.sas import SimulationActionState


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


    def apply_skill(self, skill_name, agent_id, target_object_id, callback=None):
        target_object = self.objects.get_object_by_id(target_object_id)

        sas = SimulationActionState(
            pre_container=self.objects,
            robot=agent_id,
            target_object=target_object,
            controller=self.controller,
            action_callback=callback,
            skill_name=skill_name,
        )

        result_sas = target_object.skill_portfolio.apply(sas)
        self.transations.append(result_sas.post_container)

        print(self.transations[-2].diff(self.transations[-1]))


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
                        multi_agent_event = self.controller.step(action="MoveAhead", agentId=act['agent_id'])

                    elif act['action'] == 'MoveBack':
                        multi_agent_event = self.controller.step(action="MoveBack", agentId=act['agent_id'])

                    elif act['action'] == 'RotateLeft':
                        multi_agent_event = self.controller.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])

                    elif act['action'] == 'RotateRight':
                        multi_agent_event = self.controller.step(action="RotateRight", degrees=act['degrees'],
                                                   agentId=act['agent_id'])

                    elif act["action"] == "LookUp":
                        multi_agent_event = self.controller.step(action="LookUp", agentId=act["agent_id"])

                    elif act["action"] == "LookDown":
                        multi_agent_event = self.controller.step(action="LookDown", agentId=act["agent_id"])

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
                        self.apply_skill('PutObject', agent_id=act['agent_id'], target_object_id=act['objectId'], callback=PutObjectCallback)

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
                        self.total_exec += 1
                        multi_agent_event = self.controller.step(action="SliceObject", objectId=act['objectId'],
                                                   agentId=act['agent_id'], forceAction=True)
                        if multi_agent_event.metadata['errorMessage'] != "":
                            print(multi_agent_event.metadata['errorMessage'])
                        else:
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
                        multi_agent_event = self.controller.step(action="Done")


                except Exception as e:
                    print(e)

                for i, e in enumerate(multi_agent_event.events):
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
