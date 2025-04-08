import os
import subprocess
import sys
import threading  # noqa
import time  # noqa
from glob import glob
from typing import Tuple

import numpy as np
from scipy.spatial import distance

def thread_exception_handler(args):
    print(f"Unhandled exception in thread: {args.exc_value}")
    sys.exit(1)

threading.excepthook = thread_exception_handler

def generate_video():
    frame_rate = 5
    # input_path, prefix, char_id=0, image_synthesis=['normal'], frame_rate=5
    cur_path = os.path.dirname(__file__) + "/*/"
    for imgs_folder in glob(cur_path, recursive=False):
        view = imgs_folder.split('/')[-2]
        if not os.path.isdir(imgs_folder):
            print("The input path: {} you specified does not exist.".format(imgs_folder))
        else:
            command_set = ['ffmpeg', '-i',
                           '{}/img_%05d.png'.format(imgs_folder),
                           '-framerate', str(frame_rate),
                           '-pix_fmt', 'yuv420p',
                           '{}/video_{}.mp4'.format(os.path.dirname(__file__), view)]
            subprocess.call(command_set)

>>> FILL IN SETUP CODE HERE <<< # noqa

no_robot = len(robots)
print(scene_name)

from SMARTLLM.smartllm.utils.get_controller import get_sim
from hippo.llmqueries.llm import set_api_key

api_key_path = __file__.split("smartllm")[0] + "api_key"
set_api_key(api_key_path)
simulator = get_sim(scene_name)
simulator.log_dir = tmp_hippo_log_dir
simulator.set_task_description(abstract_task_prompt)

>>> FILL IN PLAN CODE HERE <<<  # noqa

simulator.Done()
time.sleep(5)

generate_video()