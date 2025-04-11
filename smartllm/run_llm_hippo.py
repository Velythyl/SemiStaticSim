import argparse
import dataclasses
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import openai

from SMARTLLM.smartllm.utils.get_controller import get_list_of_objects
from SMARTLLM.smartllm.utils.resolve_scene import resolve_scene_id
from llmqueries import LLM
from hippo.utils.file_utils import get_tmp_folder

sys.path.append(".")

import resources.actions as actions
import resources.robots as robots


def set_api_key(openai_api_key):
    openai.api_key = Path(openai_api_key + '.txt').read_text()

# Function returns object list with name and properties.
def convert_to_dict_objprop(objs, obj_mass):
    objs_dict = []
    for i, obj in enumerate(objs):
        obj_dict = {'name': obj , 'mass' : obj_mass[i]}
        # obj_dict = {'name': obj , 'mass' : 1.0}
        objs_dict.append(obj_dict)
    return objs_dict

TARGET_TMP_DIR = get_tmp_folder()




@dataclasses.dataclass
class SceneTask:
    test_tasks: tuple
    robots_test_tasks: tuple
    gt_test_tasks: tuple
    trans_cnt_tasks: tuple
    max_trans_cnt_tasks: tuple
    available_robots: tuple
    scene: Union[str, Any]

def parse_ai2thor_plan(scene, task_path):

    # read the tasks
    test_tasks = []
    robots_test_tasks = []
    gt_test_tasks = []
    trans_cnt_tasks = []
    max_trans_cnt_tasks = []
    with open(task_path, "r") as f:
        for line in f.readlines():
            if line.startswith("//"):
                continue
            test_tasks.append(list(json.loads(line).values())[0])
            robots_test_tasks.append(list(json.loads(line).values())[1])
            gt_test_tasks.append(list(json.loads(line).values())[2])
            trans_cnt_tasks.append(list(json.loads(line).values())[3])
            max_trans_cnt_tasks.append(list(json.loads(line).values())[4])

    available_robots = []
    for robots_list in robots_test_tasks:
        task_robots = []
        for i, r_id in enumerate(robots_list):
            rob = robots.robots[r_id - 1]
            # rename the robot
            rob['name'] = 'robot' + str(i + 1)
            task_robots.append(rob)
        available_robots.append(task_robots)

    return SceneTask(test_tasks=test_tasks, robots_test_tasks=robots_test_tasks, gt_test_tasks=gt_test_tasks,
                     trans_cnt_tasks=trans_cnt_tasks, max_trans_cnt_tasks=max_trans_cnt_tasks, scene=scene,
                     available_robots=available_robots)

def parse_floorplan(args_floor_plan, args_custom_task=None):
    scene = resolve_scene_id(args_floor_plan)

    if not scene.endswith(".json"):
        task_path = f"./data/{args.test_set}/{scene}.json"
        return parse_ai2thor_plan(scene, task_path)

    # custom task is of format:
    # "<text description of the task 1>", "<text description of the task 2>,... [[<int id of robot>, <int id of robot>, ...], [<int for robot>, <int id of robot>, ...]]

    if args_custom_task is None:
        task_path = args_floor_plan.split(".json")[0]+ "_TASK.json"
        with open(task_path, "r") as f:
            task_descriptions = f.read()
    else:
        assert isinstance(args_custom_task, str)
        task_descriptions = args_custom_task

    def parse_task_string(input_string):
        # Pattern to match the tasks and robot assignments
        pattern = r'^(.*?)\s*\[\[?([\d,\s]*)\]\]?$'
        match = re.match(pattern, input_string)

        if not match:
            raise ValueError("Invalid input format")

        task_descriptions_part = match.group(1).strip()
        robots_part = match.group(2)

        # Split task descriptions (handling quoted strings if needed)
        # This simple split works for your examples, but may need adjustment for more complex cases
        task_descriptions = [desc.strip() for desc in task_descriptions_part.split(',')]
        task_descriptions = [desc.strip('"').strip("'").strip() for desc in task_descriptions if desc.strip()]

        # Parse robots assignments
        robots = []
        # Check if it's a simple case like "[1,2]"
        if '[' not in robots_part and ']' not in robots_part:
            robot_ids = [int(id_str.strip()) for id_str in robots_part.split(',') if id_str.strip()]
            robots.append(robot_ids)
        else:
            # Handle nested lists like "[[1,2], [3,4]]"
            robot_groups = re.findall(r'\[([\d,\s]*)\]', robots_part)
            for group in robot_groups:
                robot_ids = [int(id_str.strip()) for id_str in group.split(',') if id_str.strip()]
                robots.append(robot_ids)

        # Special case: if we have one task but multiple robot assignments (unlikely but possible)
        if len(task_descriptions) == 1 and len(robots) > 1:
            # We'll assume each robot assignment is for the single task
            pass
        elif len(task_descriptions) != len(robots):
            raise ValueError(
                f"Mismatch between number of tasks ({len(task_descriptions)}) and robot assignments ({len(robots)})")

        return task_descriptions, robots
    tasks, robots_for_tasks = parse_task_string(task_descriptions)

    available_robots = []
    for robots_list in robots_for_tasks:
        task_robots = []
        for i, r_id in enumerate(robots_list):
            rob = robots.robots[r_id - 1]
            # rename the robot
            rob['name'] = 'robot' + str(i + 1)
            task_robots.append(rob)
        available_robots.append(task_robots)

    return SceneTask(test_tasks=tasks, robots_test_tasks=robots_for_tasks, gt_test_tasks=None, trans_cnt_tasks=None, max_trans_cnt_tasks=None, scene=scene, available_robots=available_robots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=str, default="/home/charlie/Desktop/Holodeck/hippo/sampled_scenes/115knife/in_order_0/scene.json") #"/home/charlie/Desktop/Holodeck/hippo/sampled_scenes/0/scene.json")
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument("--gpt-version", type=str, default="gpt-4",
                        choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-16k', "bbllm"])
    
    parser.add_argument("--prompt-decompse-set", type=str, default="train_task_decompose", 
                        choices=['train_task_decompose'])
    
    parser.add_argument("--prompt-allocation-set", type=str, default="train_task_allocation", 
                        choices=['train_task_allocation'])
    
    parser.add_argument("--test-set", type=str, default="final_test", 
                        choices=['final_test'])
    
    parser.add_argument("--log-results", type=bool, default=True)

    parser.add_argument("--custom-task", type=str, default="make me a snack please [1]")
    
    args = parser.parse_args()

    # todo gotta handle the floor plan loading stuff, FloorPlanN is very different compared to HippoPlanN ???? maybe use a gym-registry style thing?

    set_api_key("../api_key")
    
    if not os.path.isdir(f"./logs/"):
        os.makedirs(f"./logs/")

    scenetask = parse_floorplan(args_floor_plan=args.floor_plan, args_custom_task=args.custom_task)
        
    
    ######## Train Task Decomposition ########
        
    # prepare train decompostion demonstration for ai2thor samples
    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"
    objects_ai = f"\n\nobjects = {get_list_of_objects(scenetask.scene)}"
    prompt += objects_ai
    
    # read input train prompts
    decompose_prompt_file = open(os.getcwd() + "/data/pythonic_plans/" + args.prompt_decompse_set + ".py", "r")
    decompose_prompt = decompose_prompt_file.read()
    decompose_prompt_file.close()
    
    prompt += "\n\n" + decompose_prompt
    
    print ("Generating Decompsed Plans...")
    
    decomposed_plan = []
    for task in scenetask.test_tasks:
        curr_prompt =  f"{prompt}\n\n# Task Description: {task}"
        
        #if "gpt" not in args.gpt_version and "bbllm" not in args.gpt_version:
        #    # older gpt versions
        #    _, text = LM(curr_prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.15)
        #else:
        #    #messages = [{"role": "user", "content": curr_prompt}]
        _, text = LLM(curr_prompt, args.gpt_version, max_tokens=1300, frequency_penalty=0.0)

        decomposed_plan.append(text)
        
    print ("Generating Allocation Solution...")

    ######## Train Task Allocation - SOLUTION ########
    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"
    
    prompt_file = os.getcwd() + "/data/pythonic_plans/" + args.prompt_allocation_set + "_solution.py"
    allocated_prompt_file = open(prompt_file, "r")
    allocated_prompt = allocated_prompt_file.read()
    allocated_prompt_file.close()
    
    prompt += "\n\n" + allocated_prompt + "\n\n"
    
    allocated_plan = []
    for i, plan in enumerate(decomposed_plan):
        no_robot  = len(scenetask.available_robots[i])
        curr_prompt = prompt + plan
        curr_prompt += f"\n# TASK ALLOCATION"
        curr_prompt += f"\n# Scenario: There are {no_robot} robots available, The task should be performed using the minimum number of robots necessary. Robots should be assigned to subtasks that match its skills and mass capacity. Using your reasoning come up with a solution to satisfy all contraints."
        curr_prompt += f"\n\nrobots = {scenetask.available_robots[i]}"
        curr_prompt += f"\n{objects_ai}"
        curr_prompt += f"\n\n# IMPORTANT: The AI should ensure that the robots assigned to the tasks have all the necessary skills to perform the tasks. IMPORTANT: Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both and allocate robots based on availablitiy. "
        curr_prompt += f"\n# SOLUTION  \n"

        #if "gpt" not in args.gpt_version and "bbllm" not in args.gpt_version:
        #    # older versions of GPT
        #    _, text = LM(curr_prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.65)
        
        if "gpt-3.5" in args.gpt_version:
            # gpt 3.5 and its variants
            #messages = [{"role": "user", "content": curr_prompt}]
            _, text = LLM(curr_prompt, args.gpt_version, max_tokens=1500, frequency_penalty=0.35)
        
        else:          
            # gpt 4.0
            messages = [{"role": "system", "content": "You are a Robot Task Allocation Expert. Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both based on your reasoning. In the case of Task Allocation based on Robot Skills alone - First check if robot teams are required. Then Ensure that robot skills or robot team skills match the required skills for the subtask when allocating. Make sure that condition is met. In the case of Task Allocation based on Mass alone - First check if robot teams are required. Then Ensure that robot mass capacity or robot team combined mass capacity is greater than or equal to the mass for the object when allocating. Make sure that condition is met. In both the Task Task Allocation based on Mass alone and Task Allocation based on Skill alone, if there are multiple options for allocation, pick the best available option by reasoning to the best of your ability."},{"role": "system", "content": "You are a Robot Task Allocation Expert"},{"role": "user", "content": curr_prompt}]
            _, text = LLM(curr_prompt, args.gpt_version, max_tokens=400, frequency_penalty=0.69)

        allocated_plan.append(text)
    
    print ("Generating Allocated Code...")
    
    ######## Train Task Allocation - CODE Solution ########

    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"
    prompt += objects_ai
    
    code_plan = []

    prompt_file1 = os.getcwd() + "/data/pythonic_plans/" + args.prompt_allocation_set + "_code.py"
    code_prompt_file = open(prompt_file1, "r")
    code_prompt = code_prompt_file.read()
    code_prompt_file.close()
    
    prompt += "\n\n" + code_prompt + "\n\n"

    for i, (plan, solution) in enumerate(zip(decomposed_plan,allocated_plan)):
        curr_prompt = prompt + plan
        curr_prompt += f"\n# TASK ALLOCATION"
        curr_prompt += f"\n\nrobots = {scenetask.available_robots[i]}"
        curr_prompt += solution
        curr_prompt += f"\n# CODE Solution  \n"
        
        #if "gpt" not in args.gpt_version and "bbllm" not in args.gpt_version:
        #    # older versions of GPT
        #    _, text = LM(curr_prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
        #else:
        #    # using variants of gpt 4 or 3.5
        messages = [{"role": "system", "content": "You are a Robot Task Allocation Expert"},{"role": "user", "content": curr_prompt}]
        _, text = LLM(messages, args.gpt_version, max_tokens=1400, frequency_penalty=0.4)

        code_plan.append(text)
    
    # save generated plan
    exec_folders = []
    if args.log_results:
        line = {}
        now = datetime.now() # current date and time
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        
        for idx, task in enumerate(scenetask.test_tasks):
            task_name = "{fxn}".format(fxn = '_'.join(task.split(' ')))
            task_name = task_name.replace('\n','')
            folder_name = f"{task_name}_plans_{date_time}"
            exec_folders.append(folder_name)
            
            os.mkdir("./logs/"+folder_name)

            with open(f"./logs/{folder_name}/available_robots.json", "w") as f:
                f.write(json.dumps(scenetask.available_robots[idx], indent=2))

            with open(f"./logs/{folder_name}/scene_name.txt", "w") as f:
                f.write(args.floor_plan)

            with open(f"./logs/{folder_name}/abstract_task_prompt.txt", "w") as f:
                f.write(scenetask.test_tasks[idx])
     
            with open(f"./logs/{folder_name}/log.txt", 'w') as f:
                f.write(task)
                f.write(f"\n\nGPT Version: {args.gpt_version}")
                f.write(f"\n\nFloor Plan: {args.floor_plan}")
                f.write(f"\n{objects_ai}")
                f.write(f"\nrobots = {scenetask.available_robots[idx]}")
                f.write(f"\nground_truth = {scenetask.gt_test_tasks[idx] if scenetask.gt_test_tasks is not None else None}")
                f.write(f"\ntrans = {scenetask.trans_cnt_tasks[idx] if scenetask.trans_cnt_tasks is not None else None}")
                f.write(f"\nmax_trans = {scenetask.max_trans_cnt_tasks[idx] if scenetask.max_trans_cnt_tasks is not None else None}")

            with open(f"./logs/{folder_name}/decomposed_plan.py", 'w') as d:
                d.write(decomposed_plan[idx])
                
            with open(f"./logs/{folder_name}/allocated_plan.py", 'w') as a:
                a.write(allocated_plan[idx])
                
            with open(f"./logs/{folder_name}/code_plan.py", 'w') as x:
                x.write(code_plan[idx])
            