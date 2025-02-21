import json

from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from tqdm import tqdm

from ai2holodeck.constants import THOR_COMMIT_ID, OBJATHOR_ASSETS_DIR
from hippo.scenedata import HippoObjectPlan
from hippo.utils.file_utils import get_tmp_folder


def get_hippo_controller(scene, target_dir=None, objathor_asset_dir=OBJATHOR_ASSETS_DIR, width=1024, height=1024):
    if isinstance(scene, str) and scene.endswith(".json"):
        with open(scene, "r") as f:
            scene = json.load(f)

    if target_dir is None:
        target_dir = get_tmp_folder()

    for object in tqdm(scene["objects"], desc="Concretizing objects..."):
        HippoObjectPlan.from_holodeckdict(object).concretize(target_dir, objathor_asset_dir)

    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=target_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )
    return controller

if __name__ == "__main__":
    get_hippo_controller("./sampled_scenes/0/scene.json", OBJATHOR_ASSETS_DIR)