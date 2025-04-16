import json
import os
from pathlib import Path
import subprocess
import argparse

from cleanplanner.parse_scene import SceneTask, PlanLog
from hippo.ai2thor_hippo_controller import get_list_of_objects
from hippo.utils.subproc import run_subproc

def fix_indentation(code: str) -> str:
    fixed_lines = []
    indent = " " * 4
    in_function = False

    for line in code.splitlines():
        stripped = line.lstrip()

        # Detect function start
        if stripped.startswith("def "):
            in_function = True
            fixed_lines.append(stripped)
            continue

        # Detect exiting a function: a non-indented, non-comment line
        if not line.startswith(" ") and not stripped.startswith("def"):
            in_function = False
            fixed_lines.append(stripped)
            continue

        # Inside a function: indent comment or code
        if in_function and stripped != "":
            fixed_lines.append(indent + stripped)
        else:
            fixed_lines.append(stripped)

    return "\n".join(fixed_lines)




def compile_aithor_exec_file(cfg, plan_log: PlanLog, feedback_output_file, executable_output_dir):

    SETUP_CODE = ""

    SETUP_CODE += f"robots = {plan_log.scenetask.robots} \n"
    SETUP_CODE += f"scene_name = '{plan_log.scenetask.scene_id}'\n"
    SETUP_CODE += f"# objects = {get_list_of_objects(plan_log.scenetask.scene_id)}\n"
    SETUP_CODE += f"abstract_task_prompt = '{plan_log.scenetask.tasks[0]}'\n"

    from hippo.utils.file_utils import get_tmp_folder
    SETUP_CODE += f"tmp_hippo_log_dir = '{feedback_output_file}'\n"
    SETUP_CODE += f"api_key_path = '{cfg.paths.curdir+'/../api_key'}'\n"
    SETUP_CODE = f"""
# >>> SETUP CODE START <<<
{SETUP_CODE}
# >>> SETUP CODE END <<<
""".strip()

    try:
        EXECUTION_TEMPLATE = Path(cfg.paths.curdir + "/datasmartllm/hippo_executable_code_template.py").read_text()
    except FileNotFoundError:
        EXECUTION_TEMPLATE = Path(cfg.paths.curdir + "/smartllm/datasmartllm/hippo_executable_code_template.py").read_text()
    EXECUTION_TEMPLATE = EXECUTION_TEMPLATE.replace(">>> FILL IN SETUP CODE HERE <<< # noqa\n", f"\n{SETUP_CODE}\n")
    PLAN_CODE = plan_log.code_plan.replace("\t", "    ")
    PLAN_CODE = fix_indentation(PLAN_CODE)

    from resources.actions import ai2thor_actions_list
    for skill in ai2thor_actions_list:
        skill = skill.split(" ")[0]
        PLAN_CODE = PLAN_CODE.replace(f"{skill}(", f"simulator.{skill}(robots[0],")

    PLAN_CODE = f"""
# >>> PLAN CODE START <<<
{PLAN_CODE}
# >>> PLAN CODE END <<<
""".strip()

    EXECUTION_TEMPLATE = EXECUTION_TEMPLATE.replace(">>> FILL IN PLAN CODE HERE <<<  # noqa\n", f"\n{PLAN_CODE}\n")

    with open(f"{executable_output_dir}/executable_plan.py", 'w') as d:
        d.write(EXECUTION_TEMPLATE)
        
    return (f"{executable_output_dir}/executable_plan.py")

def run_executable_plan(ai_exec_file):
    run_subproc(["python", ai_exec_file])

#parser = argparse.ArgumentParser()
#parser.add_argument("--command", type=str, required=True)
#args = parser.parse_args()

#expt_name = args.command
#print (expt_name)
#ai_exec_file = compile_aithor_exec_file(expt_name)


#subprocess.run(["python", ai_exec_file])