from pathlib import Path
from time import sleep

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR

from hippo.conceptgraph.conceptgraph_to_hippo import get_hippos
from hippo.reconstruction.assetlookup.CLIPLookup import CLIPLookup
from hippo.reconstruction.assetlookup.TRELLISLookup import TRELLISLookup
from hippo.reconstruction.composer import SceneComposer
from hippo.utils.subproc import run_subproc
from llmqueries.llm import set_api_key


import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import hydra

HIPPO = None

from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    global HIPPO
    if HIPPO is None:
        if cfg.assetlookup.method == "CLIP":
            HIPPO = CLIPLookup(cfg, OBJATHOR_ASSETS_DIR, do_weighted_random_selection=True, similarity_threshold=28, consider_size=True)
        elif cfg.assetlookup.method == "TRELLIS":
            
            print("Starting TRELLIS server...")
            # Ensure the TRELLIS server is started in a separate process
            trellis_proc = run_subproc(f"cd /home/mila/c/charlie.gauthier/TRELLIS && source venv2/bin/activate && CUDA_VISIBLE_DEVICES=1;HF_HOME=/network/scratch/c/charlie.gauthier/hfcache python3 flaskserver.py", shell=True, immediately_return=True)
            TOTAL_WAIT_TIME = 0
            while "WARNING: This is a development server." not in trellis_proc.stdout_stderr.getvalue():
                print("Waiting for TRELLIS server to start...")
                sleep(10)
                TOTAL_WAIT_TIME += 10
                if TOTAL_WAIT_TIME > 600:
                    raise RuntimeError("TRELLIS server did not start in time! Check the logs for errors.")
            
            sleep(10)
            print("TRELLIS server started successfully.")
            HIPPO = TRELLISLookup(cfg, OBJATHOR_ASSETS_DIR, do_weighted_random_selection=True, similarity_threshold=28, consider_size=True)

    set_api_key(cfg.secrets.openai_key)
    hipporoom, objects = get_hippos(cfg.paths.scene_dir, pad=2)

    #os.makedirs(cfg.paths.out_scene_dir, exist_ok=True)
    composer = SceneComposer.create(
        cfg,
        asset_lookup=HIPPO,
        target_dir=cfg.paths.out_scene_dir,
        objectplans=objects,
        roomplan=hipporoom
    )
    print("Writing down compositions...")
    composer.write_compositions_in_order(1)

    print("Taking topdown view...")
    composer.take_topdown()
    print("Done with scene composition and topdown view.")


if __name__ == '__main__':
    main()


"""

PYTHONPATH=..:$PYTHONPATH xvfb-run -a -s "-screen 0 1400x900x24" python3 main.py



PYTHONPATH=..:$PYTHONPATH python3 main.py  --multirun hydra/launcher=sbatch +hydra/sweep=sbatch hydra.launcher.timeout_min=30  hydra.launcher.gres=gpu:0 hydra.launcher.cpus_per_task=6 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=60 hydra.launcher.partition=long-cpu  secrets=secrets_cluster  paths.conceptgraphs_dir="./datasets/concept-nodes" assetfitting=rot_and_axisscale,rot_and_aspect_fill,rot_and_aspect_fit  paths.scene_id='replica_office2_cg-detector_2025-06-10-20-34-08.445148,replica_room0_cg-detector_2025-06-10-20-16-50.243828,replica_office0_cg-detector_2025-06-10-20-25-14.795112,replica_office3_cg-detector_2025-06-10-20-40-27.070705,replica_room1_cg-detector_2025-06-10-20-04-54.511655,replica_office1_cg-detector_2025-06-10-20-29-28.683537,replica_office4_cg-detector_2025-06-10-20-45-18.828598,replica_room2_cg-detector_2025-06-10-20-21-16.981182'  paths.out_dir="./sampled_scenes/firstrun"

"""