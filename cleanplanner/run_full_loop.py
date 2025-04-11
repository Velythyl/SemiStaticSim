import json

import hydra

import random

import wandb
from omegaconf import omegaconf, OmegaConf

from cleanplanner.planner import parse_floorplan
from hippo.utils.file_utils import get_tmp_folder


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
        # mode="disabled"
    )

    cfg_yaml = OmegaConf.to_yaml(cfg)
    print(cfg_yaml)
    with open(f"{wandb.run.dir}/hydra_config.yaml", "w") as f:
        f.write(cfg_yaml)

    return run

def resolve_cfg(cfg):
    # this only exists for fixing procthor envs: we read the tasks off of the objlist thing
    scenelist = cfg.scene.scenelist
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

    cfg.scene.scenelist = new_scenelist
    cfg.scene.tasks = new_tasklist

    if cfg.scene.robots is None:
        cfg.scene.robots = [[0] for _ in range(len(new_scenelist))]


    if cfg.paths.plan == "<TMP>":
        cfg.paths.plan = get_tmp_folder()

    if cfg.paths.curdir == "<CWD>":
        cfg.paths.curdir = "/".join(__file__.split("/")[:-1])

    return cfg


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    cfg = resolve_cfg(cfg)
    wandb_init(cfg)

    scenetask = parse_floorplan(args_floor_plan=.floor_plan, args_custom_task=args.custom_task)
    set_api_key("../api_key")
    gen_plan(scenetask)


if __name__ == "__main__":