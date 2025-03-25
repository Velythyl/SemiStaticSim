import itertools
from typing import Any, Callable

import numpy as np
import open_clip
from sentence_transformers import SentenceTransformer

from ai2holodeck.generation.holodeck import confirm_paths_exist
from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
from ai2holodeck.generation.rooms import FloorPlanGenerator
from ai2holodeck.generation.utils import get_bbox_dims, get_annotations
from ai2holodeck.generation.walls import WallGenerator
from hippo.reconstruction.scenedata import HippoRoomPlan, HippoObject


class AssetLookup:

    def __init__(self, objaverse_asset_dir: str, do_weighted_random_selection: bool, similarity_threshold: float, consider_size: bool):
        confirm_paths_exist()

        self.do_weighted_random_selection = do_weighted_random_selection
        self.similarity_threshold = similarity_threshold
        self.consider_size = consider_size

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

        self.wall_generator = WallGenerator(None)

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

        scene["wall_height"] = 3
        wall_height, walls = self.wall_generator.generate_walls(scene)
        scene["wall_height"] = wall_height
        scene["walls"] = walls

        def apply_to_positions(data: Any, func: Callable[[float, float, float], dict]):
            """
            Recursively applies `func` to every {'position': {'x': float, 'y': float, 'z': float}} sub-dictionary.

            :param data: The input data (can be dict, list, or other types).
            :param func: A function that takes (x, y, z) and returns a modified dict {'x': ..., 'y': ..., 'z': ...}.
            :return: The modified data structure.
            """
            if isinstance(data, dict):
                if "position" in data and isinstance(data["position"], dict):
                    pos = data["position"]
                    if all(k in pos and isinstance(pos[k], (int, float)) for k in ["x", "y", "z"]):
                        data["position"] = func(pos["x"], pos["y"], pos["z"])

                for key in data:
                    data[key] = apply_to_positions(data[key], func)

            elif isinstance(data, list):
                for i in range(len(data)):
                    data[i] = apply_to_positions(data[i], func)

            return data

        metadata = scene["metadata"]
        maxxcoord = scene["rooms"][0]["vertices"][2][0]
        maxzcoord = scene["rooms"][0]["vertices"][2][1]
        apply_to_positions(metadata, lambda x, y, z: {"x": 0.5, "y": y, "z": 0.5})

        return scene

    def lookup_assets(self, obj: HippoObject, size_comparison_tresh=0.1):
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

        new_object = object_dict.as_holodeckdict()

        # Add the new object to the scene
        scene['objects'].append(new_object)
        return scene
