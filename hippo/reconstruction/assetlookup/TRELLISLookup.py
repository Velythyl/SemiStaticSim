import json
import os
import shutil
import uuid

import numpy as np
from PIL import Image

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

    def __init__(self, cfg, objaverse_asset_dir: str, do_weighted_random_selection: bool, similarity_threshold: float, consider_size: bool):
        self.clip_lookup = CLIPLookup(cfg, objaverse_asset_dir, do_weighted_random_selection, similarity_threshold, consider_size)
        self._TRELLIS_client = None
        self.cfg = cfg

    @property
    def TRELLIS_client(self):
        if self._TRELLIS_client is None:
            self._TRELLIS_client = Trellis3DClient(port=self.cfg.assetlookup.client_port)
        return self._TRELLIS_client


    def generate_rooms(self, scene, plan: str=None, hipporoom: HippoRoomPlan=None):
        return self.clip_lookup.generate_rooms(scene, plan, hipporoom)

    def lookup_assets(self, obj: HippoObject, size_comparison_tresh=0.1):
        image_dir = obj._cg_paths["rgb"]
        masks_dir = obj._cg_paths["mask"]

        if self.cfg.assetlookup.use_masks:
            masked_dir = image_dir.replace("rgb", "masked")
            os.makedirs(masked_dir, exist_ok=True)
            # Get list of RGB images (case-insensitive PNG check)
            rgb_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]

            for rgb_file in rgb_files:
                # Construct paths
                rgb_path = os.path.join(image_dir, rgb_file)
                mask_path = os.path.join(masks_dir, rgb_file)  # Assuming same filename

                # Check if mask exists
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found for {rgb_file}, skipping...")
                    continue

                # Load images
                rgb_img = Image.open(rgb_path)
                mask_img = Image.open(mask_path).convert('L')  # Convert mask to grayscale

                # Convert images to numpy arrays
                rgb_array = np.array(rgb_img)
                mask_array = np.array(mask_img)

                # Ensure RGB image has 3 channels (remove alpha if present)
                if rgb_array.shape[2] == 4:
                    rgb_array = rgb_array[:, :, :3]

                # Create RGBA image by adding alpha channel
                rgba_array = np.zeros((rgb_array.shape[0], rgb_array.shape[1], 4), dtype=np.uint8)
                rgba_array[:, :, :3] = rgb_array  # Copy RGB channels

                # Set alpha channel: 255 where mask is white, 0 where black
                # Assuming white in mask is > 127 (adjust threshold if needed)
                rgba_array[:, :, 3] = (mask_array > 127) * 255

                # Set RGB values to 0 where mask is black (optional for cleaner transparency)
                rgba_array[mask_array <= 127] = [0, 0, 0, 0]

                # Convert back to PIL Image and save
                result_img = Image.fromarray(rgba_array, 'RGBA')
                output_path = os.path.join(masked_dir, rgb_file)
                result_img.save(output_path)
                print(f"Processed {rgb_file}")

            image_dir = masked_dir



        if cache.cache_says_trellis_fails(image_dir):
            print(f"TRELLIS pipeline can't handle <{image_dir}>, falling back to OBJAVERSE...")
            return self.clip_lookup.lookup_assets(obj, size_comparison_tresh=size_comparison_tresh)

        # cache lookup
        IS_FOUND_IN_CACHE = cache.is_in_cache(image_dir)
        def mark_as_infeasible(reason):
            print(f"Marking as infeasible: <{image_dir}>")

            trellisfailure_folder = cache.path_in_cache_for_trellisfailure(image_dir)
            os.makedirs(trellisfailure_folder, exist_ok=True)

            with open(f"{trellisfailure_folder}/trellis_failure.txt", "w") as f:
                f.write(
                    "If the directory <trellis_failure> exists, the creation of this asset failed. Fall back to OBJAVERSE.")
                f.write("\n")
                f.write(reason)
            return

        def obtain_and_put_in_cache():
            cache.clear_cache(image_dir)

            raw_folder = cache.path_in_cache_for_raw(image_dir)
            os.makedirs(raw_folder, exist_ok=True)

            try:
                self.TRELLIS_client.generate_and_download_from_multiple_images(
                    image_dir,
                    target_dir=raw_folder,
                    params={
                        'multiimage_algo': 'stochastic',
                        'seed': 123,
                        'preprocess_image': str(not self.cfg.assetlookup.use_masks)
                    }
                )
            except Exception as e:
                try:
                    self.TRELLIS_client.generate_and_download_from_single_image(
                        image_dir + "/000.png",
                        target_dir=raw_folder,
                        params={
                            'multiimage_algo': 'stochastic',
                            'seed': 123,
                            'preprocess_image': str(not self.cfg.assetlookup.use_masks)
                        }
                    )
                except Exception as e:
                    return mark_as_infeasible("Failed to apply TRELLIS to these images, for some reason.")

            convert_folder = cache.path_in_cache_for_convert(image_dir)
            os.makedirs(convert_folder, exist_ok=True)

            assetid = str(uuid.uuid4().hex)[:8]
            conversion_success = convert_to_ai2thor.convert(raw_folder, assetid, convert_folder)
            if not conversion_success:
                return mark_as_infeasible("Failed to convert to AI2THOR")

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

        print(f"Looking up assets for <{image_dir}>...")
        if IS_FOUND_IN_CACHE:
            print("Cache hit! Not querying TRELLIS.")
        else:
            print("Cache miss! Querying TRELLIS, this will take a while...")
            obtain_and_put_in_cache()

        if cache.cache_says_trellis_fails(image_dir):
            print(f"TRELLIS pipeline can't handle <{image_dir}>, falling back to OBJAVERSE...")
            return self.clip_lookup.lookup_assets(obj, size_comparison_tresh=size_comparison_tresh)

        with open(f"{cache.path_in_cache_for_metadata(image_dir)}/bbox.json", "r") as f:
            bbox = json.load(f)
            size = bbox["x"], bbox["y"], bbox["z"]

        with open(f"{cache.path_in_cache_for_metadata(image_dir)}/uuid.txt", "r") as f:
            assetid = f.read().strip()

        sizes = np.array(size)[None]

        return obj.add_asset_info_([assetid], sizes, [1], [cache.path_in_cache_for_convert(image_dir)])
