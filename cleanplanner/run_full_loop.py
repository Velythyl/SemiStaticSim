import json
import os
import shutil
import uuid
from pathlib import Path

import hydra

import random

import wandb
from omegaconf import omegaconf, OmegaConf

from cleanplanner.execute_plan import compile_aithor_exec_file, run_executable_plan
from cleanplanner.parse_scene import parse_floorplan, SceneTask, PlanLog
from cleanplanner.planner import gen_plan
from hippo.simulation.singlefilelog import get_last_plan_feedback
from hippo.utils.file_utils import get_tmp_folder, get_tmp_file
from llmqueries.llm import set_api_key, LLM


def set_seed(cfg, meta_key="meta"):
    seed = cfg[meta_key]["seed"]
    if seed == -1:
        seed = random.randint(0, 20000)
        cfg[meta_key]["seed"] = seed

def wandb_init(cfg, meta_key="meta"):
    set_seed(cfg,meta_key)

    run = wandb.init(
        # entity=cfg.wandb.entity,
        project=cfg[meta_key].project,
        name=cfg[meta_key]["run_name"],  # todo
        save_code=True,
        settings=wandb.Settings(start_method="thread", code_dir=".."),
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        tags=cfg[meta_key]["tags"],
        mode="disabled"# if cfg[meta_key].disable else "enabled"
    )

    cfg_yaml = OmegaConf.to_yaml(cfg)
    print(cfg_yaml)
    with open(f"{wandb.run.dir}/hydra_config.yaml", "w") as f:
        f.write(cfg_yaml)

    return run

def resolve_cfg(cfg):
    # this only exists for fixing procthor envs: we read the tasks off of the objlist thing
    scenelist = cfg.scene.sceneids
    scenetasks = cfg.scene.tasks

    with open("../objlist_tasked.json", 'r') as f:
        procthor_task_dict = json.load(f)

    new_scenelist = []
    new_tasklist = []
    for i, (scene, tasks) in enumerate(zip(scenelist, scenetasks)):
        if "procthor" in scene.lower():
            assert tasks is None
            scene_id = scene.replace("procthor", '')
            assert str(int(scene_id)) == scene_id
            tasks = procthor_task_dict[scene_id]["tasks"]
            assert tasks is not None

        new_tasklist.append(tasks)
        new_scenelist.append(scene)

    new_tasklist = [(t if isinstance(t, list) else [t]) for t in new_tasklist]

    _new_tasklist = []  # fixme maybe instead of this complicated thing, just have multi-tasks be a set of separated strings, like "make a meal|clean the knife"
    for sc in new_tasklist:
        temp = []
        for each in sc:
            if isinstance(each, str):
                each = [each]
            temp.append(each)
        _new_tasklist.append(temp)
    new_tasklist = _new_tasklist

    cfg.scene.sceneids = new_scenelist
    cfg.scene.tasks = new_tasklist

    if cfg.scene.robots is None:
        cfg.scene.robots = [[1] for _ in range(len(new_scenelist))]
    cfg.scene.robots = [(r if isinstance(r, list) else [r]) for r in cfg.scene.robots]

    if cfg.paths.curdir == "<CWD>":
        cfg.paths.curdir = "/".join(__file__.split("/")[:-1])

    return cfg

def run_scenetask(cfg, scenetask: SceneTask, num_retries: int = 0, feedback: str = None, run_id:str = "blank_run_id"):
    print("run_id", run_id)
    cfg.meta.run_name = scenetask.sc_ta_ro_ID
    wandb_init(cfg)

    NUM_INPUT_TOKENS, NUM_OUTPUT_TOKENS = 0, 0

    plan_output_dir = get_tmp_folder()
    plan_log = gen_plan(cfg, scenetask, plan_output_dir, feedback)
    NUM_INPUT_TOKENS += plan_log.num_input_tokens
    NUM_OUTPUT_TOKENS += plan_log.num_output_tokens

    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    hydra_cfg = HydraConfig.get()
    scene_name = OmegaConf.to_container(hydra_cfg.runtime.choices)["scene"]
    cwd = "/".join(__file__.split("/")[:-1])
    local_plan_output_dir = f"{cwd}/planresults/{scene_name}/{scenetask.tasks[0].replace('.', ' ').replace(' ', '-')}/{cfg.planner.llm}/{run_id}"
    os.makedirs(local_plan_output_dir, exist_ok=True)

    log_file = get_tmp_file()
    executable_output_dir = get_tmp_folder()
    executable_path = compile_aithor_exec_file(cfg, plan_log, log_file, executable_output_dir)
    print(f"executing: {executable_path}")
    run_executable_plan(executable_path)
    print(f"reading log file: {log_file}")


    from hippo.simulation.singlefilelog import is_plan_success, get_necessary_plan_feedback
    num_tries_so_far = len(os.listdir(local_plan_output_dir))
    local_plan_output_dir = f"{local_plan_output_dir}/{num_tries_so_far}"
    os.makedirs(local_plan_output_dir, exist_ok=False)
    shutil.copy(log_file, f"{local_plan_output_dir}/log_fie.json")
    with open(f"{local_plan_output_dir}/plan_log.json", "w") as f:
        json.dump(plan_log.asdict(), f, indent=4)
    shutil.copy(f"{executable_output_dir}/output.mp4", f"{local_plan_output_dir}/sim.mp4")
    with open(f"{local_plan_output_dir}/was_success.txt", "w") as f:
        f.write(f"{is_plan_success(log_file)}")
    with open(f"{local_plan_output_dir}/necessary_plan_feedback.txt", "w") as f:
        f.write(f"{get_necessary_plan_feedback(log_file)}")
    with open(f"{local_plan_output_dir}/plan.txt", "w") as f:
        f.write(plan_log.code_plan)


    last_feedback = get_last_plan_feedback(log_file)
    necessary_feedback = get_necessary_plan_feedback(log_file)

    WANDB_LOG = {
        "execute/success": int(is_plan_success(log_file)),
        "planning/input_tokens_this_round": plan_log.num_input_tokens,
        "planning/output_tokens_this_round": plan_log.num_output_tokens,
        "planning/total_input_tokens": NUM_INPUT_TOKENS,
        "planning/total_output_tokens": NUM_OUTPUT_TOKENS,
        "planning/code_plan": plan_log.code_plan,
        "execute/last_feedback": get_last_plan_feedback(log_file),
        "execute/last_feedback_type": get_last_plan_feedback(log_file)["Type"],
        "execute/last_feedback_message": get_last_plan_feedback(log_file)["Error message"],
        "execute/full_log_file": Path(log_file).read_text()
    }

    if get_necessary_plan_feedback(log_file) is not None:
        WANDB_LOG.update({
            "execute/feedback": get_necessary_plan_feedback(log_file),
            "execute/feedback_type": get_necessary_plan_feedback(log_file)["Type"],
            "execute/feedback_message": get_necessary_plan_feedback(log_file)["Error message"]
        })

    wandb.log(WANDB_LOG)

    IS_SUCCESS = is_plan_success(log_file)
    if IS_SUCCESS:
        return True
    else:
        if ask_if_plan_can_be_fixed(plan_log, get_necessary_plan_feedback(log_file)):
            feedback = f'# ------\n\n# Last time you tried to generate a plan for this task and scene, you generated the following:\n"""{plan_log.code_plan}"""\n\n# But the plan failed, and you got this error message:\n#{necessary_feedback["Error message"]}\n# ------'
            print("run_id", run_id)
            return run_scenetask(cfg, scenetask, num_retries+1, feedback, run_id=run_id)
        else:
            print("Plan cannot be fixed, aborting.")

    exit()

def ask_if_plan_can_be_fixed(plan_log: PlanLog, necessary_plan_feedback: dict):
    PROMPT = f"""
We detected that this plan was faulty:

TASK DESCRIPTION: {plan_log.scenetask.tasks[0]}
PLAN: {plan_log.code_plan}

Due to the following reasons: 

TYPE: {necessary_plan_feedback['Type']}
MESSAGE: {necessary_plan_feedback['Error message']}

In your opinion, is it worth it to go for another round of LLM planning? Or is the plan just infeasible.
Note: you cannot change the TASK DESCRIPTION, and any plan must obey the TASK DESCRIPTION (but the contents of the plan might change).
If the TASK DESCRIPTION contains explicit instructions, any plan will also contain them.

Output format:

Reasoning:

Is this plan satisfiable?:

Does this plan violate LLM alignment constraints?:

Final output decision: 

Output `APPROVE REPLAN` anywhere in your answer to replan and try to generate a new plan for that task.
Output `CANCEL REPLAN` anywhere in your answer to abort and terminate the program.
"""
    _, response = LLM(PROMPT, "gpt-4")

    if "APPROVE REPLAN" in response:
        return True
    if "CANCEL REPLAN" in response:
        return False
    raise AssertionError()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    cfg = resolve_cfg(cfg)

    run_id = uuid.uuid4().hex
    set_api_key("../hippo/secrets/openai_api_key")
    for scene_id, tasks_for_scene_id, robots_in_scene_id in zip(cfg.scene.sceneids, cfg.scene.tasks, cfg.scene.robots):
        for task_for_scene_id, robot_for_scene_id in zip(tasks_for_scene_id, robots_in_scene_id):
            scenetask = parse_floorplan(scene_id, task_for_scene_id, robot_for_scene_id)
            run_scenetask(cfg, scenetask, run_id=run_id)


if __name__ == "__main__":
    main()