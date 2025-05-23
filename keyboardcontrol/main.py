import cv2
import gzip
import json
import math
import os
from typing import Sequence
from tqdm import tqdm
import numpy as np


actionList = {
    "MoveAhead": "w",
    "MoveBack": "s",
    "MoveLeft": "a",
    "MoveRight": "d",
    "RotateLeft": "q",
    "RotateRight": "e",
    "LookUp": "r",
    "LookDown": "f",
    "Record": "t",
    "FINISH": "p",
    # Interact action
    "PickupObject": "z",
    "PutObject": "x",
    "OpenObject": "c",
    "CloseObject": "v",
    "ToggleObjectOn": "b",
    "ToggleObjectOff": "n",
    "SliceObject": "m",
}


def get_interact_object(env, action, pickup=None):
    candidates = []
    objectId = ''


    for obj in env.last_event.metadata["objects"]:
        if obj["visible"]:
            candidates.append(obj["objectId"])
    if len(candidates) == 0:
        print('no valid interact object candidates')
        return None
    else:
        print('===========choose index from the candidates==========')
        for index, obj in enumerate(candidates):
            print(index, ':', obj)
        while True:
            # input the index of candidates in the console
            keystroke = input()
            print(keystroke)
            if keystroke == actionList["FINISH"]:
                print("stop interact")
                break
            try:
                objectId = candidates[int(keystroke)]
            except:
                print("INVALID KEY", keystroke)
                continue
            print(objectId)
            break
        return objectId

def keyboard_play(env, top_down_frames, first_view_frames, is_rotate, rotate_per_frame):
    first_view_frame = env.last_event.frame
    cv2.imshow("first_view", cv2.cvtColor(first_view_frame, cv2.COLOR_RGB2BGR))

    # remove the ceiling
    env.step(action="ToggleMapView")
    top_down_frame = env.last_event.third_party_camera_frames[0]
    cv2.imshow("top_view", cv2.cvtColor(top_down_frame, cv2.COLOR_RGB2BGR))
    env.step(action="ToggleMapView")

    step = 0

    while True:
        keystroke = cv2.waitKey(0)
        step += 1

        if keystroke == ord(actionList["FINISH"]):
            env.stop()
            cv2.destroyAllWindows()
            print("action: STOP")
            break

        if keystroke == ord(actionList["MoveAhead"]):
            action = "MoveAhead"
            print("action: MoveAhead")
        elif keystroke == ord(actionList["MoveBack"]):
            action = "MoveBack"
            print("action: MoveBack")
        elif keystroke == ord(actionList["MoveLeft"]):
            action = "MoveLeft"
            print("action: MoveLeft")
        elif keystroke == ord(actionList["MoveRight"]):
            action = "MoveRight"
            print("action: MoveRight")
        elif keystroke == ord(actionList["RotateLeft"]):
            action = "RotateLeft"
            print("action: RotateLeft")
        elif keystroke == ord(actionList["RotateRight"]):
            action = "RotateRight"
            print("action: RotateRight")
        elif keystroke == ord(actionList["LookUp"]):
            action = "LookUp"
            print("action: LookUp")
        elif keystroke == ord(actionList["LookDown"]):
            action = "LookDown"
            print("action: LookDown")

        elif keystroke == ord(actionList["PickupObject"]):
            action = "PickupObject"
            objectId = get_interact_object(env, action)
            pickup = objectId.split('|')[0]
            print("action: PickupObject")
        elif keystroke == ord(actionList["PutObject"]):
            action = "PutObject"
            print('holding', pickup)
            objectId = get_interact_object(env, action, pickup=pickup)
            print("action: PutObject")
        elif keystroke == ord(actionList["OpenObject"]):
            action = "OpenObject"
            objectId = get_interact_object(env, action)
            print("action: OpenObject")
        elif keystroke == ord(actionList["CloseObject"]):
            action = "CloseObject"
            objectId = get_interact_object(env, action)
            print("action: CloseObject")
        elif keystroke == ord(actionList["ToggleObjectOn"]):
            action = "ToggleObjectOn"
            objectId = get_interact_object(env, action)
            print("action: ToggleObjectOn")
        elif keystroke == ord(actionList["ToggleObjectOff"]):
            action = "ToggleObjectOff"
            objectId = get_interact_object(env, action)
            print("action: ToggleObjectOff")
        elif keystroke == ord(actionList["SliceObject"]):
            action = "SliceObject"
            objectId = get_interact_object(env, action)
            print("action: SliceObject")
        else:
            print("INVALID KEY", keystroke)
            continue

        # agent step
        if "Object" in action:
            env.step(action=action, objectId=objectId)
        else:
            env.step(action=action)

        if is_rotate:
            ## rotation third camera
            pose = compute_rotate_camera_pose(env.last_event.metadata["sceneBounds"]["center"],
                                              env.last_event.metadata["thirdPartyCameras"][0], rotate_per_frame)

            env.step(
                action="UpdateThirdPartyCamera",
                **pose
            )

        first_view_frame = env.last_event.frame
        cv2.imshow("first_view", cv2.cvtColor(first_view_frame, cv2.COLOR_RGB2BGR))

        # remove the ceiling
        env.step(action="ToggleMapView")
        top_down_frame = env.last_event.third_party_camera_frames[0]
        cv2.imshow("top_view", cv2.cvtColor(top_down_frame, cv2.COLOR_RGB2BGR))
        env.step(action="ToggleMapView")

        top_down_frames.append(top_down_frame)
        first_view_frames.append(first_view_frame)


def show_video(frames: Sequence[np.ndarray], fps: int = 10):
    """Show a video composed of a sequence of frames.

    Example:
    frames = [
        controller.step("RotateRight", degrees=5).frame
        for _ in range(72)
    ]
    show_video(frames, fps=5)
    """
    frames = ImageSequenceClip(frames, fps=fps)
    return frames


def export_video(path, frames):
    """Merges all the saved frames into a .mp4 video and saves it to `path`"""

    video = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        5,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        # assumes that the frames are RGB images. CV2 uses BGR.
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.destroyAllWindows()
    video.release()


def compute_rotate_camera_pose(center, pose, degree_per_frame=6):
    """degree_per_frame: set the degree of rotation for each frame"""

    def rotate_pos(x1, y1, x2, y2, degree):
        angle = math.radians(degree)
        n_x1 = (x1 - x2) * math.cos(angle) - (y1 - y2) * math.sin(angle) + x2
        n_y1 = (x1 - x2) * math.sin(angle) + (y1 - y2) * math.cos(angle) + y2

        return n_x1, n_y1

    # print(math.sqrt((pose["position"]["x"]-center["x"])**2 + (pose["position"]["z"]-center["z"])**2))
    x, z = rotate_pos(pose["position"]["x"], pose["position"]["z"], center["x"], center["z"], degree_per_frame)
    pose["position"]["x"], pose["position"]["z"] = x, z

    direction_x = center["x"] - x
    direction_z = center["z"] - z
    pose["rotation"]["y"] = math.degrees(math.atan2(direction_x, direction_z))

    # print(math.sqrt((pose["position"]["x"]-center["x"])**2 + (pose["position"]["z"]-center["z"])**2))

    return pose


def initialize_side_camera_pose(scene_bound, pose, third_fov=60, slope_degree=45, down_angle=70, scale_factor=8):
    """
    down_angle: the x-axis rotation angle of the camera, represents the top view of the front view from top to bottom, which needs to be less than 90 degrees
    ensure the line vector between scene's center & camera 's angel equal down_angle
    scale_factor scale the camera's view, make it larger ensure camera can see the whole scene
    """
    fov_rad = np.radians(third_fov)
    pitch_rad = np.radians(down_angle)
    distance = (scene_bound["center"]["y"] / 2) / np.tan(fov_rad / 2)
    pose["position"]["y"] = scene_bound["center"]["y"] + distance * scale_factor * np.sin(pitch_rad)
    pose["position"]["z"] = scene_bound["center"]["z"] - distance * scale_factor * np.cos(pitch_rad)

    pose["rotation"]["x"] = down_angle

    pose["orthographic"] = False
    del pose["orthographicSize"]

    pose = compute_rotate_camera_pose(scene_bound["center"], pose, slope_degree)

    return pose


def main(scene_name="FloorPlan205_physics", gridSize=0.25, rotateStepDegrees=15,
         BEV=False, slope_degree=45, down_angle=65, use_procthor=False, procthor_scene_file="", procthor_scene_num=100,
         is_rotate=True, rotate_per_frame=6, generate_video=False, generate_gif=False):
    ## procthor room
    if use_procthor:
        with gzip.open(procthor_scene_file, "r") as f:
            houses = [line for line in tqdm(f, total=10000, desc=f"Loading train")]
        ## procthor train set's room
        house = json.loads(houses[procthor_scene_num])
    else:
        ## select room, 1-30，201-230，301-330，401-430 are ithor's room
        house = scene_name

    from SMARTLLM.smartllm.utils.resolve_scene import resolve_scene_id

    #floor_no = "/home/charlie/Desktop/Holodeck/SMARTLLM/hipposcenes/9/scene.json"
    #scene = resolve_scene_id(floor_no)
    #from SMARTLLM.smartllm.utils.get_controller import get_controller
    #controller = get_controller(scene, get_runtime_container=False, width=1000, height=1000, snapToGrid=False,
    #                                      visibilityDistance=100, fieldOfView=90, gridSize=0.25, rotateStepDegrees=20)

    from hippo.ai2thor_hippo_controller import get_sim
    controller = get_sim(scene_name, just_controller=True)
    """
    _ = Controller(
        agentMode="default",
        visibilityDistance=5,
        renderInstanceSegmentation=True,
        scene=house,
        # step sizes
        gridSize=gridSize,
        snapToGrid=False,
        rotateStepDegrees=rotateStepDegrees,
        # camera properties
        width=1200,
        height=800,
        fieldOfView=90,
        platform=CloudRendering,
    )
    """

    ## add third view camera
    event = controller.step(action="GetMapViewCameraProperties")
    ## third camera's fov
    third_fov = 60

    if not BEV:
        ## top_view(slope)
        pose = initialize_side_camera_pose(event.metadata["sceneBounds"], event.metadata["actionReturn"], third_fov,
                                           slope_degree, down_angle)
    else:
        ## BEV
        pose = event.metadata["actionReturn"]
        is_rotate = False  ## assume that BEV do not need rotation

    event = controller.step(
        action="AddThirdPartyCamera",
        skyboxColor="black",
        fieldOfView=third_fov,
        **pose
    )

    ## collect frame
    first_view_frames = []
    third_view_frames = []

    ## use keyboard control agent
    keyboard_play(controller, third_view_frames, first_view_frames, is_rotate, rotate_per_frame)

    ## use frames generate video
    if generate_video:

        if not os.path.exists("./video"):
            os.mkdir("./video")

        export_video("./video/first_view_{}.mp4".format(scene_name), first_view_frames)
        export_video("./video/third_view_{}.mp4".format(scene_name), third_view_frames)

    ## use frames generate gif
    if generate_gif:

        if not os.path.exists("./gif"):
            os.mkdir("./gif")

        clip = show_video(third_view_frames, fps=5)
        clip.write_gif("./gif/third_view_{}.gif".format(scene_name))
        clip2 = show_video(first_view_frames, fps=5)
        clip2.write_gif("./gif/first_view_{}.gif".format(scene_name))


if __name__ == "__main__":
    main(scene_name="/home/charlie/Desktop/Holodeck/hippo/sampled_scenes/replica_room0_cg-detector_2025-04-04-18-03-58/119/in_order_0/scene.json",  # FloorPlan19_physics ## room
         gridSize=0.25, rotateStepDegrees=15,  ## agent step len and rotate degree
         BEV=False,  ## Bird's-eye view or top view(slope)
         slope_degree=60,  ## top view(slope)'s initial rotate degree
         down_angle=65,  ## top view(slope)'s pitch angle, should be 0-90, 90 equal to Bird's-eye view
         use_procthor=False,  ## use procthor room, True: select room from procthor train set, need dataset dir
         procthor_scene_file="",  ## procthor train set dir
         procthor_scene_num=100,  ## select scene from procthor train set
         is_rotate=True,  ## top_view rotate?
         rotate_per_frame=6,  ## top_view rotate degree
         generate_video=True,  ## use frames generate video
         generate_gif=True,  ## use frames generate gif
         )