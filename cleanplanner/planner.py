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

from cleanplanner.parse_scene import SceneTask, PlanLog
from hippo.ai2thor_hippo_controller import get_list_of_objects
from hippo.utils.file_utils import get_tmp_folder
from llmqueries.llm import LLM, approx_num_tokens

sys.path.append(".")

import resources.actions as actions
import resources.robots as robots


def set_api_key(openai_api_key):
    openai.api_key = Path(openai_api_key + '.txt').read_text()


# Function returns object list with name and properties.
def convert_to_dict_objprop(objs, obj_mass):
    objs_dict = []
    for i, obj in enumerate(objs):
        obj_dict = {'name': obj, 'mass': obj_mass[i]}
        # obj_dict = {'name': obj , 'mass' : 1.0}
        objs_dict.append(obj_dict)
    return objs_dict


TARGET_TMP_DIR = get_tmp_folder()





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




def gen_plan(cfg, scenetask: SceneTask, output_dir):
    assert os.path.exists(output_dir)

    ######## Train Task Decomposition ########

    # prepare train decompostion demonstration for ai2thor samples
    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"

    # read input train prompts
    decompose_prompt = Path(cfg.paths.curdir + "/datasmartllm/pythonic_plans/" + cfg.planner.prompt + ".py").read_text()

    prompt += "\n\n" + decompose_prompt

    print("Generating Decompsed Plans...")

    objects_ai = f"\n\nobjects = {get_list_of_objects(scenetask.scene)}"
    prompt += f"\n\n### NOW, CONSIDER THIS SITUATION:{objects_ai}"
    curr_prompt = f"{prompt}\n\n# Task Description: {scenetask.tasks[0]}"

    NUM_INPUT_TOKENS = approx_num_tokens(cfg.planner.llm, curr_prompt)
    _, decomposed_plan = LLM(curr_prompt, cfg.planner.llm, max_tokens=1300, frequency_penalty=0.0)
    NUM_OUTPUT_TOKENS = approx_num_tokens(cfg.planner.llm, decomposed_plan)

    print("Plan obtained! Saving...")
    """
    ######## Train Task Allocation - SOLUTION ########
    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"

    allocated_prompt = Path(cfg.paths.curdir + "/datasmartllm/pythonic_plans/" + cfg.planner.prompt_allocation_set + "_solution.py").read_text()

    prompt += "\n\n" + allocated_prompt + "\n\n"


    no_robot = len(scenetask.robots)
    curr_prompt = prompt + decomposed_plan
    curr_prompt += f"\n# TASK ALLOCATION"
    curr_prompt += f"\n# Scenario: There are {no_robot} robots available, The task should be performed using the minimum number of robots necessary. Robots should be assigned to subtasks that match its skills and mass capacity. Using your reasoning come up with a solution to satisfy all contraints."
    curr_prompt += f"\n\nrobots = {scenetask.robots}"
    curr_prompt += f"\n{objects_ai}"
    curr_prompt += f"\n\n# IMPORTANT: The AI should ensure that the robots assigned to the tasks have all the necessary skills to perform the tasks. IMPORTANT: Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both and allocate robots based on availablitiy. "
    curr_prompt += f"\n# SOLUTION  \n"

    if "gpt-3.5" in cfg.planner.llm:
        _, allocated_plan = LLM(curr_prompt, cfg.planner.llm, max_tokens=1500, frequency_penalty=0.35)

    else:
        # gpt 4.0
        messages = [{"role": "system",
                     "content": "You are a Robot Task Allocation Expert. Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both based on your reasoning. In the case of Task Allocation based on Robot Skills alone - First check if robot teams are required. Then Ensure that robot skills or robot team skills match the required skills for the subtask when allocating. Make sure that condition is met. In the case of Task Allocation based on Mass alone - First check if robot teams are required. Then Ensure that robot mass capacity or robot team combined mass capacity is greater than or equal to the mass for the object when allocating. Make sure that condition is met. In both the Task Task Allocation based on Mass alone and Task Allocation based on Skill alone, if there are multiple options for allocation, pick the best available option by reasoning to the best of your ability."},
                    #{"role": "system", "content": "You are a Robot Task Allocation Expert"},
                    {"role": "user", "content": curr_prompt}]
        _, allocated_plan = LLM(messages, cfg.planner.llm, max_tokens=400, frequency_penalty=0.69)

    print("Generating Allocated Code...")

    ######## Train Task Allocation - CODE Solution ########

    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"
    prompt += objects_ai


    code_prompt = Path(cfg.paths.curdir + "/datasmartllm/pythonic_plans/" + cfg.planner.prompt_allocation_set + "_code.py").read_text()

    prompt += "\n\n" + code_prompt + "\n\n"

    curr_prompt = prompt + decomposed_plan
    curr_prompt += f"\n# TASK ALLOCATION"
    curr_prompt += f"\n\nrobots = {scenetask.robots}"
    curr_prompt += allocated_plan
    curr_prompt += f"\n# CODE Solution  \n"

    messages = [{"role": "system", "content": "You are a Robot Task Allocation Expert"},
                {"role": "user", "content": curr_prompt}]
    _, code_plan = LLM(messages, cfg.planner.llm, max_tokens=1400, frequency_penalty=0.4)

    """

    # save generated plan
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


    plan_log = PlanLog(
        scenetask = scenetask,
        llm= cfg.planner.llm,
        num_input_tokens=NUM_INPUT_TOKENS,
        num_output_tokens=NUM_OUTPUT_TOKENS,
        llm_outputs={"code_plan": decomposed_plan},
        code_plan=decomposed_plan,
    )



    with open(f"{output_dir}/plan_log.json", "w") as f:
        f.write(json.dumps(plan_log.asdict(), indent=2, sort_keys=True))
    return plan_log

    with open(f"{output_dir}/available_robots.json", "w") as f:
        f.write(json.dumps(scenetask.robots, indent=2))

    with open(f"{output_dir}/scene_name.txt", "w") as f:
        f.write(scenetask.scene_id)

    with open(f"{output_dir}/abstract_task_prompt.txt", "w") as f:
        f.write(scenetask.tasks)

    with open(f"{output_dir}/log.txt", 'w') as f:
        f.write(task)
        f.write(f"\n\nGPT Version: {args.gpt_version}")
        f.write(f"\n\nFloor Plan: {args.floor_plan}")
        f.write(f"\n{objects_ai}")
        f.write(f"\nrobots = {scenetask.available_robots[idx]}")
        f.write(
            f"\nground_truth = {scenetask.gt_test_tasks[idx] if scenetask.gt_test_tasks is not None else None}")
        f.write(
            f"\ntrans = {scenetask.trans_cnt_tasks[idx] if scenetask.trans_cnt_tasks is not None else None}")
        f.write(
            f"\nmax_trans = {scenetask.max_trans_cnt_tasks[idx] if scenetask.max_trans_cnt_tasks is not None else None}")

    #with open(f"./logs/{folder_name}/decomposed_plan.py", 'w') as d:
    #    d.write(decomposed_plan[idx])

    #with open(f"./logs/{folder_name}/allocated_plan.py", 'w') as a:
    #    a.write(allocated_plan[idx])

    #with open(f"./logs/{folder_name}/code_plan.py", 'w') as x:
    #    x.write(code_plan[idx])
