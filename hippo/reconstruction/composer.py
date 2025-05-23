import copy
import dataclasses
import itertools
import json
import os
import random
from typing import Tuple, Dict, List, Union

import numpy as np
from tqdm import tqdm

from ai2holodeck.generation.utils import get_top_down_frame
from hippo.ai2thor_hippo_controller import get_hippo_controller
from hippo.reconstruction.assetlookup._AssetLookup import AssetLookup
from hippo.reconstruction.llm_annotation import LLM_annotate
from hippo.reconstruction.scenedata import HippoObject, HippoRoomPlan
from hippo.utils.selfdataclass import SelfDataclass
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

with open("../ai2holodeck/generation/empty_house.json", "r") as f:
    DEFAULT_SCENE = json.load(f)

@dataclasses.dataclass
class SceneComposer(SelfDataclass):
    cfg: DictConfig
    asset_lookup: AssetLookup
    target_dir: str
    objectplans: Tuple[HippoObject]
    roomplan: HippoRoomPlan
    scene: Dict

    @classmethod
    def create(cls, cfg, asset_lookup: AssetLookup, target_dir: str, objectplans: Tuple[HippoObject], roomplan: HippoRoomPlan, KEEP_TOP_K: int = 3):
        looked_up_objectplans = []
        for obj in objectplans:
            looked_up_obj = asset_lookup.lookup_assets(obj)[:KEEP_TOP_K]

            if cfg.skillprediction.method == None:
                looked_up_obj2 = looked_up_obj
            else:
                looked_up_obj2 = LLM_annotate(cfg, looked_up_obj)

            looked_up_objectplans.append(looked_up_obj2)



        #objectplans = tuple([asset_lookup.lookup_assets(x,asset_lookup.objaverse_asset_dir)[:KEEP_TOP_K] for x in objectplans])
        scene = asset_lookup.generate_rooms(DEFAULT_SCENE, hipporoom=roomplan)

        return cls(
            cfg=cfg,
            asset_lookup=asset_lookup,
            target_dir=target_dir,
            objectplans=looked_up_objectplans,
            roomplan=roomplan,
            scene=scene
        )

    @property
    def done_paths(self):
        return list(map(lambda x: f"{self.target_dir}/{x}", os.listdir(self.target_dir)))

    @property
    def asset_dir(self):
        return self.asset_lookup.objaverse_asset_dir

    @property
    def _object_indices(self):
        ret = []
        for obj in self.objectplans:
            ret.append(list(np.arange(len(obj))))
        return ret

    @property
    def _object_indices_prod(self):
        product = list(itertools.product(*self._object_indices))
        return product

    def __len__(self):
        return len(self._object_indices_prod)

    def generate_compositions_in_order(self, prod=None, MAX_NUM=10):
        if prod is None:
            prod = self._object_indices_prod

        COUNTER = 0
        for possible_scene in prod:
            new_objectplans = []
            for i, obj_i in enumerate(possible_scene):
                new_objectplans.append(self.objectplans[i][obj_i])
            yield self.replace(objectplans=tuple(new_objectplans))
            COUNTER += 1
            if COUNTER >= MAX_NUM:
                break

    def _write_compositions(self, generator, generationname):
        COUNTER = 0
        for selves in generator:
            WRITE_DIR = f"{self.target_dir}/{generationname}{COUNTER}"
            os.makedirs(WRITE_DIR, exist_ok=False)
            CONCRETIZATION_DIR = f"{WRITE_DIR}/concrete_assets"
            os.makedirs(CONCRETIZATION_DIR, exist_ok=False)

            scene = selves.get_scene(CONCRETIZATION_DIR)
            with open(f"{self.target_dir}/{generationname}{COUNTER}/scene.json", "w") as f:
                json.dump(scene, f, indent=4)

            with open(f"{self.target_dir}/{generationname}{COUNTER}/cfg.yaml", "w") as f:
                OmegaConf.save(config=self.cfg, f=f)

            COUNTER += 1

    def write_compositions_in_order(self, MAX_NUM=10):
        self._write_compositions(self.generate_compositions_in_order(MAX_NUM=MAX_NUM), "in_order_")

    def generate_random_compositions(self, MAX_NUM=10):
        prod = self._object_indices_prod

        randprod = []
        for obj_indices in prod:
            random.shuffle(obj_indices)
            randprod.append(obj_indices)

        yield from self.generate_compositions_in_order(prod, MAX_NUM)

    def write_random_compositions(self, MAX_NUM=10):
        return self._write_compositions(self.generate_random_compositions(MAX_NUM=MAX_NUM), "random_")

    def generate_most_likely_composition(self):
        objs = []
        for obj in self.objectplans:
            i = np.argmax(np.array(obj._found_scores))
            objs.append(obj[i])

        return self.replace(objectplans=tuple(objs))

    def write_most_likely_composition(self):
        return self._write_compositions(self.generate_most_likely_composition(), "most_likely_")

    def get_scene(self, concrete_asset_dir):
        scene = copy.deepcopy(self.scene)

        for obj in tqdm(self.objectplans, desc="Concretizing objects..."):

            if len(obj) > 1:
                obj = obj[0]

            obj.concretize(concrete_asset_dir)

            # Ensure the 'objects' list exists in the scene
            if 'objects' not in scene:
                scene['objects'] = []

            new_object = obj.as_holodeckdict()

            # Add the new object to the scene
            scene['objects'].append(new_object)

        return scene

    def take_topdown(self, paths: Union[List[str],str]=None):
        if paths is None:
            paths = self.done_paths
        if isinstance(paths, str):
            paths = [paths]

        todo_paths = []
        for path in paths:
            if path.endswith(".json"):
                todo_paths.append(path)
            else:
                todo_paths.append(path + "/scene.json")

        for path in todo_paths:
            controller = get_hippo_controller(path)
            top_image = get_top_down_frame(controller, 1024, 1024)

            savepath = "/".join(path.split("/")[:-1]) + "/topdown.png"
            top_image.save(savepath)
