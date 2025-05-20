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
from hippo.reconstruction.assetlookup.CLIPLookup import CLIPLookup
from hippo.reconstruction.assetlookup.TRELLISUtils.flaskclient import Trellis3DClient
from hippo.reconstruction.scenedata import HippoRoomPlan, HippoObject

DEFAULT_THOR_METADATA = {       # doesn't actually matter much; gets overwritten when we use LLM skill annotations
    "assetMetadata": {
        "primaryProperty": "CanPickup",
        "secondaryProperties": [
            "Receptacle"
        ]
    }
}


class TRELLISLookup:

    def __init__(self, objaverse_asset_dir: str, do_weighted_random_selection: bool, similarity_threshold: float, consider_size: bool):
        self.clip_lookup = CLIPLookup(objaverse_asset_dir, do_weighted_random_selection, similarity_threshold, consider_size)
        self.TRELLIS_client = Trellis3DClient()

    def generate_rooms(self, scene, plan: str=None, hipporoom: HippoRoomPlan=None):
        return self.clip_lookup.generate_rooms(scene, plan, hipporoom)

    def lookup_assets(self, obj: HippoObject, size_comparison_tresh=0.1):
        image_dir = obj["paths"]["rgb"]

        from hippo.utils.file_utils import get_tmp_folder
        target_folder = get_tmp_folder()
        self.TRELLIS_client.generate_and_download_from_multiple_images(
            image_dir,
            target_dir=target_folder,
            params={
                'multiimage_algo': 'stochastic',
                'seed': 123
            }
        )



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
