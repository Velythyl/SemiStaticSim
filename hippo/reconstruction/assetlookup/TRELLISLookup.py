import json
import os
import shutil
import uuid

import numpy as np

from hippo.reconstruction.assetlookup.CLIPLookup import CLIPLookup
from hippo.reconstruction.assetlookup.TRELLISUtils import convert_to_ai2thor, cache
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
        image_dir = obj._cg_paths["rgb"]


        # cache lookup
        IS_FOUND_IN_CACHE = cache.is_in_cache(image_dir)
        def obtain_and_put_in_cache():
            cache.clear_cache(image_dir)

            raw_folder = cache.path_in_cache_for_raw(image_dir)
            os.makedirs(raw_folder, exist_ok=True)

            self.TRELLIS_client.generate_and_download_from_multiple_images(
                image_dir,
                target_dir=raw_folder,
                params={
                    'multiimage_algo': 'stochastic',
                    'seed': 123
                }
            )

            convert_folder = cache.path_in_cache_for_convert(image_dir)
            os.makedirs(convert_folder, exist_ok=True)

            assetid = str(uuid.uuid4().hex)[:8]
            convert_to_ai2thor.convert(raw_folder, assetid, convert_folder)

            from ai2thor.util.runtime_assets import load_existing_thor_asset_file
            obj = load_existing_thor_asset_file(convert_folder, f"{assetid}/{assetid}")
            from hippo.utils.spatial_utils import get_ai2thor_object_bbox
            bbox = get_ai2thor_object_bbox(obj)

            metadata_folder = cache.path_in_cache_for_metadata(image_dir)
            os.makedirs(metadata_folder, exist_ok=True)

            with open(f"{metadata_folder}/bbox.json", "w") as f:
                bbox = {k: float(v) for k, v in bbox.items()}
                json.dump(bbox, f, indent=2)

            with open(f"{metadata_folder}/uuid.txt", "w") as f:
                f.write(assetid)

        if IS_FOUND_IN_CACHE:
            print("Cache hit! Not querying TRELLIS.")
        else:
            print("Cache miss! Querying TRELLIS, this will take a while...")
            obtain_and_put_in_cache()

        with open(f"{cache.path_in_cache_for_metadata(image_dir)}/bbox.json", "r") as f:
            bbox = json.load(f)
            size = bbox["x"], bbox["y"], bbox["z"]

        with open(f"{cache.path_in_cache_for_metadata(image_dir)}/uuid.txt", "r") as f:
            assetid = f.read().strip()

        sizes = np.array(size)[None]

        return obj.add_asset_info_([assetid], sizes, [1], cache.path_in_cache_for_convert(image_dir))
