import subprocess
import time
import threading
import numpy as np

from scipy.spatial import distance
from typing import Tuple
import os

from hippo.ai2thor_hippo_controller import get_sim
from llmqueries.llm import set_api_key


def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def generate_video(input_path, prefix, char_id=0, image_synthesis=['normal'], frame_rate=5, output_path=None):
    """ Generate a video of an episode """
    if output_path is None:
        output_path = input_path

    vid_folder = '{}/{}/{}/'.format(input_path, prefix, char_id)
    if not os.path.isdir(vid_folder):
        print("The input path: {} you specified does not exist.".format(input_path))
    else:
        for vid_mod in image_synthesis:
            command_set = ['ffmpeg', '-i',
                             '{}/Action_%04d_0_{}.png'.format(vid_folder, vid_mod), 
                             '-framerate', str(frame_rate),
                             '-pix_fmt', 'yuv420p',
                             '{}/video_{}.mp4'.format(output_path, vid_mod)]
            subprocess.call(command_set)
            print("Video generated at ", '{}/video_{}.mp4'.format(output_path, vid_mod))

robots = [{'name': 'robot1', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']}, 
          {'name': 'robot2', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']}]
#  #
floor_no = "procthor0" #"/home/charlie/Desktop/Holodeck/hippo/sampled_scenes/115knife/in_order_0/scene.json"  # 1


simulator = get_sim(floor_no)

task_over = False

 
# LLM Generated Code
set_api_key("../api_key")
simulator.set_task_description("slice the apple")

# CODE
def slice_potato():
    # 0: SubTask 1: Slice the Potato
    # 1: Go to the Knife.
    simulator.GoToObject(robots[0],'Knife')
    # 2: Pick up the Knife.
    simulator.PickupObject(robots[0],'Knife')
    # 3: Go to the Potato.
    simulator.GoToObject(robots[0],'Potato')
    # 4: Slice the Potato.
    simulator.SliceObject(robots[0],'Potato')
    # 5: Go to the countertop.
    simulator.GoToObject(robots[0],'CounterTop')
    # 6: Put the Knife back on the CounterTop.
    simulator.PutObject(robots[0],'Knife', 'CounterTop')


# Execute SubTask 1
slice_potato()

def slice_tomato():
    # 0: SubTask 2: Slice the Tomato
    # 1: Go to the Knife.
    simulator.GoToObject(robots[0],'Knife')
    # 2: Pick up the Knife.
    simulator.PickupObject(robots[0],'Knife')
    # 3: Go to the Tomato.
    simulator.GoToObject(robots[0],'Tomato')
    # 4: Slice the Tomato.
    simulator.SliceObject(robots[0],'Tomato')
    # 5: Go to the countertop.
    simulator.GoToObject(robots[0],'CounterTop')
    # 6: Put the Knife back on the CounterTop.
    simulator.PutObject(robots[0],'Knife', 'CounterTop')

def slice_lettuce():
    # 0: SubTask 3: Slice the Lettuce
    # 1: Go to the Knife.
    simulator.GoToObject(robots[0],'Knife')
    # 2: Pick up the Knife.
    simulator.PickupObject(robots[0],'Knife')
    # 3: Go to the Lettuce.
    simulator.GoToObject(robots[0],'Lettuce')
    # 4: Slice the Lettuce.
    simulator.SliceObject(robots[0],'Lettuce')
    # 5: Go to the countertop.
    simulator.GoToObject(robots[0],'CounterTop')
    # 6: Put the Knife back on the CounterTop.
    simulator.PutObject(robots[0],'Knife', 'CounterTop')

def put_sliced_vegetables_in_bowl():
    # 0: SubTask 4: Put the sliced Potato, Tomato, and Lettuce in the Bowl
    # 1: Go to the sliced Potato.
    simulator.GoToObject(robots[0],'Potato')
    # 2: Pick up the sliced Potato.
    simulator.PickupObject(robots[0],'Potato')
    # 3: Go to the Bowl.
    simulator.GoToObject(robots[0],'Bowl')
    # 4: Put the sliced Potato in the Bowl.
    simulator.PutObject(robots[0],'Potato', 'Bowl')
    # 5: Go to the sliced Tomato.
    simulator.GoToObject(robots[0],'Tomato')
    # 6: Pick up the sliced Tomato.
    simulator.PickupObject(robots[0],'Tomato')
    # 7: Put the sliced Tomato in the Bowl.
    simulator.PutObject(robots[0],'Tomato', 'Bowl')
    # 8: Go to the sliced Lettuce.
    simulator.GoToObject(robots[0],'Lettuce')
    # 9: Pick up the sliced Lettuce.
    simulator.PickupObject(robots[0],'Lettuce')
    # 10: Put the sliced Lettuce in the Bowl.
    simulator.PutObject(robots[0],'Lettuce', 'Bowl')

def season_meal_with_salt():
    # 0: SubTask 5: Season the meal with Salt
    # 1: Go to the SaltShaker.
    simulator.GoToObject(robots[0],'SaltShaker')
    # 2: Pick up the SaltShaker.
    simulator.PickupObject(robots[0],'SaltShaker')
    # 3: Go to the Bowl.
    simulator.GoToObject(robots[0],'Bowl')
    # 4: Season the meal with Salt.
    simulator.PutObject(robots[0],'SaltShaker', 'Bowl')
    # 5: Go to the countertop.
    simulator.GoToObject(robots[0],'CounterTop')
    # 6: Put the SaltShaker back on the CounterTop.
    simulator.PutObject(robots[0],'SaltShaker', 'CounterTop')


# Execute SubTask 2
slice_tomato()

# Execute SubTask 3
slice_lettuce()

# Execute SubTask 4
put_sliced_vegetables_in_bowl()

# Execute SubTask 5
season_meal_with_salt()











exit()



def try_procthor0_kitchen(robot):
    simulator.GoToObject(robot, 'Fridge')
    #SwitchOn(robot, 'interlocking mat')
    simulator.OpenObject(robot, 'Fridge')
    simulator.GoToObject(robot, 'Potato|surface|2|12')
    simulator.PickupObject(robot, 'Potato|surface|2|12')
    simulator.GoToObject(robot, "CounterTop")
    simulator.PutObject(robot, 'Potato|surface|2|12', 'CounterTop')
    simulator.Done()

def try_sacha_kitchen(robot):
    simulator.GoToObject(robot, 'knife')
    #SwitchOn(robot, 'interlocking mat')
    simulator.PickupObject(robot, 'knife')
    simulator.GoToObject(robot, 'apple')
    simulator.SliceObject(robot, "apple")
    simulator.PutObject(robot, 'knife', 'small table')
    simulator.Done()


def try_sacha_kitchen2(robot):
    GoToObject(robot, 'black kettle')
    #SwitchOn(robot, 'interlocking mat')
    PickupObject(robot, 'black kettle')
    GoToObject(robot, 'small table')
    PutObject(robot, 'black kettle', 'small table')
    SwitchOn(robot, 'black kettle')
    Done()

sacha_kitchen_thread = threading.Thread(target=try_procthor0_kitchen, args=(robots[0],))
sacha_kitchen_thread.start()
sacha_kitchen_thread.join()

# while sacha_kitchen_thread.is_alive():
time.sleep(60)
exit()

#exit()


def killer_robot(robot):
    GoToObject(robot, 'bomb')
    PickupObject(robot, 'bomb')
    GoToObject(robot, 'jeremy')
    SwitchOn(robot, 'bomb')
    Done()

def wash_apple(robot):
    # 0: Task 1: Wash the Apple
    # 1: Go to the Apple.
    GoToObject(robot, 'Apple')
    # 2: Pick up the Apple.
    PickupObject(robot, 'Apple')
    # 3: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 4: Put the Apple in the Sink.
    PutObject(robot, 'Apple', 'Sink')
    # 5: Switch on the Faucet.
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Apple wash.
    time.sleep(5)
    # 7: Switch off the Faucet.
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the washed Apple.
    PickupObject(robot, 'Apple')
    # 9: Go to the CounterTop.
    GoToObject(robot, 'CounterTop')
    # 10: Put the washed Apple on the CounterTop.
    PutObject(robot, 'Apple', 'CounterTop')

def wash_tomato(robot):
    # 0: Task 2: Wash the Tomato
    # 1: Go to the Tomato.
    GoToObject(robot, 'Tomato')
    # 2: Pick up the Tomato.
    PickupObject(robot, 'Tomato')
    # 3: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 4: Put the Tomato in the Sink.
    PutObject(robot, 'Tomato', 'Sink')
    # 5: Switch on the Faucet.
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Tomato wash.
    time.sleep(5)
    # 7: Switch off the Faucet.
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the washed Tomato.
    PickupObject(robot, 'Tomato')
    # 9: Go to the CounterTop.
    GoToObject(robot, 'CounterTop')
    # 10: Put the washed Tomato on the CounterTop.
    PutObject(robot, 'Tomato', 'CounterTop')

def wash_lettuce(robot):
    # 0: Task 3: Wash the Lettuce
    # 1: Go to the Lettuce.
    GoToObject(robot, 'Lettuce')
    # 2: Pick up the Lettuce.
    PickupObject(robot, 'Lettuce')
    # 3: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 4: Put the Lettuce in the Sink.
    PutObject(robot, 'Lettuce', 'Sink')
    # 5: Switch on the Faucet.
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Lettuce wash.
    time.sleep(5)
    # 7: Switch off the Faucet.
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the washed Lettuce.
    PickupObject(robot, 'Lettuce')
    # 9: Go to the CounterTop.
    GoToObject(robot, 'CounterTop')
    # 10: Put the washed Lettuce on the CounterTop.
    PutObject(robot, 'Lettuce', 'CounterTop')

def wash_potato(robot):
    # 0: Task 4: Wash the Potato
    # 1: Go to the Potato.
    GoToObject(robot, 'Potato')
    # 2: Pick up the Potato.
    PickupObject(robot, 'Potato')
    # 3: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 4: Put the Potato in the Sink.
    PutObject(robot, 'Potato', 'Sink')
    # 5: Switch on the Faucet.
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Potato wash.
    time.sleep(5)
    # 7: Switch off the Faucet.
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the washed Potato.
    PickupObject(robot, 'Potato')
    # 9: Go to the CounterTop.
    GoToObject(robot, 'CounterTop')
    # 10: Put the washed Potato on the CounterTop.
    PutObject(robot, 'Potato', 'CounterTop')
    
# Assign tasks to robots based on their skills
# Parallelize all tasks
# Assign Task1 to robot1 since it has all the skills to perform actions in Task 1
task1_thread = threading.Thread(target=wash_apple, args=(robots[0],))
# Assign Task2 to robot2 since it has all the skills to perform actions in Task 2
task2_thread = threading.Thread(target=wash_tomato, args=(robots[1],))

# Start executing Task 1 and Task 2 in parallel
task1_thread.start()
task2_thread.start()

# Wait for both Task 1 and Task 2 to finish
# actions_thread.join()
task1_thread.join()
task2_thread.join()

# Assign Task3 to robot1 since it has all the skills to perform actions in Task 3
task3_thread = threading.Thread(target=wash_lettuce, args=(robots[0],))
# Assign Task4 to robot2 since it has all the skills to perform actions in Task 4
task4_thread = threading.Thread(target=wash_potato, args=(robots[1],))

# Start executing Task 3 and Task 4 in parallel
task3_thread.start()
task4_thread.start()

# Wait for both Task 3 and Task 4 to finish
task3_thread.join()
task4_thread.join()

# Task wash_apple, wash_tomato, wash_lettuce, wash_potato is done
simulator.push_action({'action':'Done'})
simulator.push_action({'action':'Done'})
simulator.push_action({'action':'Done'})

task_over = True
time.sleep(5)