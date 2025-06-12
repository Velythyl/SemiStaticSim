from pathlib import Path

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR

from hippo.conceptgraph.conceptgraph_to_hippo import get_hippos
from hippo.reconstruction.assetlookup.CLIPLookup import CLIPLookup
from hippo.reconstruction.assetlookup.TRELLISLookup import TRELLISLookup
from hippo.reconstruction.composer import SceneComposer
from llmqueries.llm import set_api_key


def get_target_dir(dataset_name, target_dir="./sampled_scenes"):
    target_dir = os.path.join(target_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    runid = len(os.listdir(target_dir))

    TARGET_DIR = f"{target_dir}/{runid}"
    os.makedirs(TARGET_DIR, exist_ok=True)
    return TARGET_DIR

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import hydra

HIPPO = None

from omegaconf import OmegaConf
OmegaConf.register_new_resolver(
    "load_file", lambda filename: Path(filename).read_text()
)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    global HIPPO
    if HIPPO is None:
        if cfg.assetlookup.method == "CLIP":
            HIPPO = CLIPLookup(cfg, OBJATHOR_ASSETS_DIR, do_weighted_random_selection=True, similarity_threshold=28, consider_size=True)
        elif cfg.assetlookup.method == "TRELLIS":
            HIPPO = TRELLISLookup(cfg, OBJATHOR_ASSETS_DIR, do_weighted_random_selection=True, similarity_threshold=28, consider_size=True)

    hipporoom, objects = get_hippos(cfg.paths.scene_dir, pad=2)
    set_api_key(cfg.secrets.openai_key)

    with open(f"{cfg.paths.out_scene_dir}/hydra_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    composer = SceneComposer.create(
        cfg,
        asset_lookup=HIPPO,
        target_dir=cfg.paths.out_scene_dir,
        objectplans=objects,
        roomplan=hipporoom
    )
    composer.write_compositions_in_order(1)

    composer.take_topdown()


if __name__ == '__main__':
    main()


"""

PYTHONPATH=..:$PYTHONPATH xvfb-run -a -s "-screen 0 1400x900x24" python3 main.py

"""