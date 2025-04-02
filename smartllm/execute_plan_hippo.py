import json
import os
from pathlib import Path
import subprocess
import argparse

def append_trans_ctr(allocated_plan):
    brk_ctr = 0
    code_segs = allocated_plan.split("\n\n")
    fn_calls = []
    for cd in code_segs:
        if "def" not in cd and "threading.Thread" not in cd and "join" not in cd and cd[-1] == ")":
            # fn_calls.append(cd)
            brk_ctr += 1
    print ("No Breaks: ", brk_ctr)
    return brk_ctr

def compile_aithor_exec_file(expt_name):
    log_path = expt_name #os.getcwd() + "/logs/" + expt_name
    executable_plan = ""
    
    # append the imports to the file
    #import_file = Path(os.getcwd() + "/data/aithor_connect/imports_aux_fn_hippo.py").read_text()
    #executable_plan += (import_file + "\n")
    
    # append the list of robots and floor plan number
    #log_file = open(log_path + "/log.txt")
    #log_data = log_file.readlines()

    SETUP_CODE = ""

    # append the robot list
    with open(log_path + "/available_robots.json", "r") as f:
        robot_list = json.dumps(json.load(f))
    SETUP_CODE += f"robots = {robot_list} \n"
    SETUP_CODE += f"scene_name = '{Path(log_path + '/scene_name.txt').read_text()}'\n"
    SETUP_CODE += f"abstract_task_prompt = '{Path(log_path + '/abstract_task_prompt.txt').read_text()}'\n"

    
    # append the ai thoe connector and helper fns
    #connector_file = Path(os.getcwd() + "/data/aithor_connect/aithor_connect.py").read_text()
    #executable_plan += (connector_file + "\n")
    
    # append the allocated plan
    #brks = append_trans_ctr(allocated_plan)
    #executable_plan += (allocated_plan + "\n")
    #executable_plan += ("no_trans = " + str(brks) + "\n")

    # append the task thread termination
    #terminate_plan = Path(os.getcwd() + "/data/aithor_connect/end_thread.py").read_text()
    #executable_plan += (terminate_plan + "\n")

    EXECUTION_TEMPLATE = Path(os.getcwd() + "/data/hippo_executable_code_template.py").read_text()
    EXECUTION_TEMPLATE = EXECUTION_TEMPLATE.replace(">>> FILL IN SETUP CODE HERE <<< # noqa\n", f"\n{SETUP_CODE}\n")
    PLAN_CODE = Path(log_path + "/code_plan.py").read_text()

    from smartllm.resources.actions import ai2thor_actions_list
    for skill in ai2thor_actions_list:
        skill = skill.split(" ")[0]
        PLAN_CODE = PLAN_CODE.replace(skill, f"simulator.{skill}")

    EXECUTION_TEMPLATE = EXECUTION_TEMPLATE.replace(">>> FILL IN PLAN CODE HERE <<<  # noqa\n", f"\n{PLAN_CODE}\n")

    with open(f"{log_path}/executable_plan.py", 'w') as d:
        d.write(EXECUTION_TEMPLATE)
        
    return (f"{log_path}/executable_plan.py")

parser = argparse.ArgumentParser()
parser.add_argument("--command", type=str, required=True)
args = parser.parse_args()

expt_name = args.command
print (expt_name)
ai_exec_file = compile_aithor_exec_file(expt_name)

subprocess.run(["python", ai_exec_file])