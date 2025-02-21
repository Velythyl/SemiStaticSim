import json
import os
import shutil

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from ai2holodeck.generation.utils import get_top_down_frame
from hippo.assetlookup import Hippo
from hippo.conceptgraph.conceptgraph_to_hippo import get_hippos
from hippo.composer import ObjectComposer
from hippo.utils.file_utils import get_tmp_folder

if __name__ == '__main__':

    hippo = Hippo(OBJATHOR_ASSETS_DIR, do_weighted_random_selection=True, similarity_threshold=28, consider_size=True)

    with open("../ai2holodeck/generation/empty_house.json", "r") as f:
        scene = json.load(f)

    hipporoom, objects = get_hippos("./rgbd_interactions_2_l14")
    print(hipporoom.coords)

    KEEP_TOP_K = 3
    new_scene = hippo.generate_rooms(scene, hipporoom.asholodeckstr())
    objects = [hippo.lookup_assets(obj)[:KEEP_TOP_K] for obj in objects]

    composer = ObjectComposer(target_dir=get_tmp_folder(), objectplans=objects, scene=new_scene, asset_dir=OBJATHOR_ASSETS_DIR)
    new_scene = composer.get_scene()

    shutil.rmtree("./sampled_scenes")
    os.makedirs("./sampled_scenes", exist_ok=True)
    for i, sampled_scene in enumerate(composer.generate_compositions_in_order()):
        os.makedirs(f"./sampled_scenes/{i}", exist_ok=True)
        top_image = get_top_down_frame(sampled_scene, composer.target_dir, 1024, 1024)
        top_image.save(f"./sampled_scenes/{i}/topdown.png")
        with open(f"./sampled_scenes/{i}/scene.json", "w") as f:
            json.dump(new_scene, f, indent=4)
        break
    exit()


    objects = [obj[0] for obj in objects]
    for obj in tqdm(objects, desc="Concretizing..."):
        obj.concretize(target_dir, OBJATHOR_ASSETS_DIR)

    for obj in objects:
        #print(obj.object_name)
        #print(obj.position)
        #print(obj._selected_size)
        obj = obj.replace(position=(obj.position[0], obj.position[1], obj.position[2]))
        #obj["position"] = (obj["position"][0], obj["position"][1], 0)
        new_scene = hippo.add_object(new_scene, obj)


    with open("./temp.json", "w") as f:
        json.dump(new_scene, f, indent=4)

    top_image = get_top_down_frame(new_scene, target_dir, 1024, 1024)
    top_image.show()
    exit()
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