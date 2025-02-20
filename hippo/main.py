import dataclasses
import itertools
import json
import os
import random
import struct
import uuid
from typing import Tuple, Dict, Any, List, Union

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from ai2holodeck.generation.holodeck import confirm_paths_exist
from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
from ai2holodeck.generation.rooms import FloorPlanGenerator
from ai2holodeck.generation.utils import get_annotations, get_bbox_dims, get_top_down_frame, room_video
from hippo.conceptgraph_to_hippo import get_hippos
from hippo.flaxdataclass import selfdataclass, SelfDataclass
from hippo.scenedata import HippoRoomPlan, HippoObjectPlan


class Hippo:

    def __init__(self, objaverse_asset_dir: str, do_weighted_random_selection: bool, similarity_threshold: float, consider_size: bool, quantity_per_object):
        confirm_paths_exist()

        self.do_weighted_random_selection = do_weighted_random_selection
        self.similarity_threshold = similarity_threshold
        self.consider_size = consider_size
        #self.quantity_per_object = quantity_per_object

        (
            self.clip_model,
            _,
            self.clip_preprocess,
        ) = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

        # initialize sentence transformer
        self.sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

        # objaverse version and asset dir
        self.objaverse_asset_dir = objaverse_asset_dir
        self.object_size_tolerance = 0.8

        # initialize generation
        self.retrieval_threshold = 28
        self.object_retriever = ObjathorRetriever(
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,
            clip_tokenizer=self.clip_tokenizer,
            sbert_model=self.sbert_model,
            retrieval_threshold=self.retrieval_threshold,
        )
        self.database = self.object_retriever.database

        self.floor_generator = FloorPlanGenerator(
            self.clip_model, self.clip_preprocess, self.clip_tokenizer, None
        )



    def generate_rooms(self, scene, plan: str=None, hipporoom: HippoRoomPlan=None):
        if not plan:
            assert hipporoom is not None
            plan = hipporoom.asholodeckstr()
        if not hipporoom:
            assert plan is not None

        rooms = self.floor_generator.get_plan("programmatic floor query", plan)
        scene["rooms"] = rooms
        return scene

    def compute_size_difference(self, target_size, candidates):

        candidate_sizes = []
        for i, (uid, score) in enumerate(candidates):
            size = get_bbox_dims(self.database[uid])
            size_list = [size["x"], size["y"], size["z"]]
            candidates[i] = (uid, score, tuple(size_list))
            candidate_sizes.append(size_list)

        candidate_sizes = torch.tensor(candidate_sizes)

        target_size_list = list(target_size)
        target_size = torch.tensor(target_size_list)

        size_difference = abs(candidate_sizes - target_size).mean(axis=1) / 100
        size_difference = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(candidates):
            candidates_with_size_difference.append(
                (uid, score - size_difference[i] * 10)
            )

        # sort the candidates by score
        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        return candidates_with_size_difference



    def lookup_assets(self, obj: HippoObjectPlan, size_comparison_tresh=0.1):

        candidates = self.object_retriever.retrieve(
            [f"a 3D model of {obj.object_name}, {obj.object_description}"],
            self.similarity_threshold,
        )

        candidates = [
            candidate
            for candidate, annotation in zip(
                candidates,
                [
                    get_annotations(self.database[candidate[0]])
                    for candidate in candidates
                ],
            )
            if all(  # ignore doors and windows and frames
                k not in annotation["category"].lower()
                for k in ["door", "window", "frame"]
            )
        ]

        def get_asset_size(uid):
            size = get_bbox_dims(self.database[uid])
            return size["x"], size["y"], size["z"]

        uids, scores = list(zip(*candidates))
        sizes = [get_asset_size(uid) for uid in uids]


        sizes = np.array(sizes)
        size_comparizon = np.linalg.norm(sizes - np.array(obj._desired_size), axis=1)
        size_ok = size_comparizon < size_comparison_tresh

        if size_ok.sum() >= 1:

            uids = list(itertools.compress(uids, size_ok))
            sizes = sizes[size_ok]
            scores = list(itertools.compress(scores, size_ok))
        else:
            print(f"could not find appropriate size for {obj.object_name}")
            best_size = size_comparizon.argmin()
            uids = [uids[best_size]]
            sizes = [sizes[best_size]]
            scores = [scores[best_size]]

        return obj.add_asset_info_(uids, sizes, scores)

        # No candidates found
        if len(candidates) == 0:
            raise Exception("TODO. Refine object query?")

        # consider object size difference
        if obj.size is not None and self.consider_size:
            candidates = self.compute_size_difference(
                obj.size, candidates
            )

        return obj.replace(_found_assetIds=tuple(candidates), assetId=candidates[0])

        #candidates = candidates[:self.quantity_per_object] # todo get top k scores, or just add scores to the tuples idk

        selected_objects = [(f"{object_type}-{i}", candidate[0]) for i, candidate in enumerate(candidates)]

        return selected_objects

    def add_object(self, scene, object_dict):
        """
        Add an object to the scene.

        Parameters:
        - scene (dict): The scene dictionary.
        - object_dict (dict): Dictionary containing object properties.
          Required keys:
            - assetId (str): The asset ID of the object.
            - position (dict): A dictionary with 'x', 'y', and 'z' coordinates.
            - roomId (str): The room where the object will be placed.
          Optional keys:
            - rotation (dict): A dictionary with 'x', 'y', and 'z' rotation values. Default is no rotation.
            - object_name (str): A name for the object. Default is based on assetId.
        """

        # Ensure the 'objects' list exists in the scene
        if 'objects' not in scene:
            scene['objects'] = []

        new_object = object_dict.asholodeckdict()

        # Add the new object to the scene
        scene['objects'].append(new_object)
        return scene

@dataclasses.dataclass
class ObjectComposer(SelfDataclass):
    objectplans: Tuple[HippoObjectPlan] = tuple()
    scene: Dict = dataclasses.field(default_factory=dict)

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


    def _add_object_(self, object_dict: Union[List[HippoObjectPlan], HippoObjectPlan]):
        """
                Add an object to the scene.

                Parameters:
                - scene (dict): The scene dictionary.
                - object_dict (dict): Dictionary containing object properties.
                  Required keys:
                    - assetId (str): The asset ID of the object.
                    - position (dict): A dictionary with 'x', 'y', and 'z' coordinates.
                    - roomId (str): The room where the object will be placed.
                  Optional keys:
                    - rotation (dict): A dictionary with 'x', 'y', and 'z' rotation values. Default is no rotation.
                    - object_name (str): A name for the object. Default is based on assetId.
                """

        # Ensure the 'objects' list exists in the scene
        if 'objects' not in scene:
            scene['objects'] = []

        new_object = object_dict.asholodeckdict()

        # Add the new object to the scene
        scene['objects'].append(new_object)
        return self.replace(scene=scene)

    def generate_compositions_in_order(self, prod=None):
        if prod is None:
            prod = self._object_indices_prod

        for possible_scene in prod:
            for i, obj_i in enumerate(possible_scene):
                self = self._add_object_(self.objectplans[i][obj_i])
            yield self.scene

    def sample_composition(self):
        prod = self._object_indices_prod

        randprod = []
        for obj in prod:
            random.shuffle(obj)
            randprod.append(obj)

        yield from self.generate_compositions_in_order(prod)






if __name__ == '__main__':

    hippo = Hippo(OBJATHOR_ASSETS_DIR, do_weighted_random_selection=True, similarity_threshold=28, consider_size=True, quantity_per_object=10)

    with open("../ai2holodeck/generation/empty_house.json", "r") as f:
        scene = json.load(f)


    #hipporoom = HippoRoomPlan("locobot scene")

    hipporoom, objects = get_hippos("./rgbd_interactions_2_l14")
    print(hipporoom.coords)

    new_scene = hippo.generate_rooms(scene, hipporoom.asholodeckstr())
    objects = [hippo.lookup_assets(obj) for obj in objects]

    composer = ObjectComposer(new_scene, objects)

    for obj in objects:
        print(obj.object_name)
        print(obj.position)
        print(obj._selected_size)
        obj = obj.replace(position=(obj.position[0], 0.1, obj.position[2]))
        #obj["position"] = (obj["position"][0], obj["position"][1], 0)
        new_scene = hippo.add_object(new_scene, obj)


    with open("./temp.json", "w") as f:
        json.dump(new_scene, f, indent=4)

    # top_image = get_top_down_frame(new_scene, OBJATHOR_ASSETS_DIR, 1024, 1024)
    # top_image.show()
    #top_image.save("./temp.png")

    final_video = room_video(scene, OBJATHOR_ASSETS_DIR, 1024, 1024, camera_height=0.3)
    final_video.write_videofile(
        "./temp.mp4", fps=30
    )
    exit()

    #object = HippoObject(id="id0", roomId=hipporoom.id, object_name="kettle", description="black electric kettle", assetId=None)


    temp = hippo.lookup_assets({"object_name": "kettle", "description": "black electric kettle", "size": (15, 20, 24)})
    object_dict = {
        "assetId": "7075f67e22524936ad00939c0ef939ed",
        "position": {"x": 5, "y": 0.1, "z": 5},
        "roomId": "living room",
    }

    object_dict2 =  {
            "assetId": "a57234d7c0f04d24af50ae6a5f1e86e9",
            "id": "sofa-0 (living room)",
            "kinematic": True,
            "position": {
                "x": 6.281492817785571,
                "y": 0.38371189935765426,
                "z": 1.75
            },
            "rotation": {
                "x": 0,
                "y": 270,
                "z": 0
            },
            "material": None,
            "roomId": "living room",
            "vertices": [
                [
                    551.7985635571142,
                    32.09263348785322
                ],
                [
                    551.7985635571142,
                    317.9073665121468
                ],
                [
                    704.5,
                    317.9073665121468
                ],
                [
                    704.5,
                    32.09263348785322
                ]
            ],
            "object_name": "sofa-0",
            "layer": "Procedural0"
        }

    new_scene = hippo.add_object(new_scene, object_dict)

    top_image = get_top_down_frame(scene, OBJATHOR_ASSETS_DIR, 1024, 1024)
    # top_image.show()
    top_image.save("./temp.png")

    i=0