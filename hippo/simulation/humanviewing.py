import dataclasses
import json
import os
import re
import sys
from typing import Union

import cv2

import numpy as np
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from langchain.llms import Anyscale
from tqdm import tqdm

from ai2holodeck.constants import THOR_COMMIT_ID, OBJATHOR_ASSETS_DIR
from hippo.simulation.runtimeobjects import RuntimeObjectContainer
from hippo.reconstruction.scenedata import HippoObject, dict2xyztuple
from hippo.utils.file_utils import get_tmp_folder

import os

import ai2thor.controller
from hippo.simulation.ai2thor_metadata_reader import get_object_list_from_controller, get_robot_inventory
from hippo.simulation.skill_simulator import Simulator

from hippo.simulation.spatialutils.filter_positions import build_grid_graph, filter_reachable_positions

from ai2holodeck.constants import THOR_COMMIT_ID
from hippo.ai2thor_hippo_controller import get_hippo_controller_OLDNOW, get_hippo_controller
from hippo.utils.file_utils import get_tmp_folder
@dataclasses.dataclass
class HumanViewing:
    c: Controller
    r: RuntimeObjectContainer
    s: Union[Simulator, None]
    plan: list[str] = dataclasses.field(default_factory=lambda : ["Cossin", "Bidule"])
    current_action_idx: int = 0   # 0 = nothing yet, len(plan)+1 = all done

    def set_plan(self, plan):
        if isinstance(plan, str):
            plan = list(filter(lambda x: len(x) > 0, map(str.strip, plan.split("\n"))))
            return plan
        return plan

    def incr_action_idx(self):
        self.current_action_idx = self.current_action_idx + 1

    def get_latest_robot_frame(self):
        first_view_frame = self.c.last_event.frame
        return first_view_frame

    def get_augmented_robot_frame(self, frame, message=None, held_item=None, hud_scale=2):
        import cv2, re
        if held_item is None:
            inventory = get_robot_inventory(self.c, 0)
            assert len(inventory) <= 1
            if len(inventory) == 0:
                held_item = None
            else:
                raw_item = inventory[0]
                held_item = re.match(r"^([^-]+)", raw_item).group(1)

        if message is None:
            if self.s is not None:
                message_queue = self.s.exception_queue
                if len(message_queue) <= 0:
                    message = None
                else:
                    message = message_queue[-1]
        if message is None or message == "":
            message = "[None yet]"

        hud_frame = frame.copy()
        h, w = hud_frame.shape[:2]

        def draw_box(img, top_left, bottom_right, color=(0, 0, 0), alpha=0.5):
            overlay = img.copy()
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
            return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        pad = int(20 * hud_scale)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 * hud_scale
        title_font_scale = 0.5 * hud_scale
        thickness = max(1, int(1 * hud_scale))

        # Calculate section widths (1/3 of the viewer each)
        section_width = w // 3
        inner_pad = pad // 2  # Padding inside each section

        # ---------------- Plan Display (Left section) ----------------
        plan_x = inner_pad
        plan_y = pad
        title_plan = "Plan:"
        title_plan_size = cv2.getTextSize(title_plan, font, title_font_scale, thickness)[0]

        # Calculate plan box dimensions (1/3 of viewer width)
        plan_box_width = section_width - 2 * inner_pad
        plan_lines = len(self.plan) if self.plan else 1
        plan_box_height = title_plan_size[1] + (plan_lines * int(20 * hud_scale)) + 2 * inner_pad

        # Draw plan box
        plan_box_tl = (plan_x, plan_y + title_plan_size[1] + inner_pad)
        plan_box_br = (plan_box_tl[0] + plan_box_width, plan_box_tl[1] + plan_box_height)
        hud_frame = draw_box(hud_frame, plan_box_tl, plan_box_br, (50, 50, 0), 0.7)  # Yellowish background

        # Draw title
        cv2.putText(hud_frame, title_plan, (plan_x, plan_y + title_plan_size[1]),
                    font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Draw plan steps
        line_y = plan_y + title_plan_size[1] + inner_pad + 35
        for idx, step in enumerate(self.plan, start=1):
            color = (180, 180, 180)  # default gray
            if self.current_action_idx == idx:
                color = (0, 255, 255)  # highlight current step
            elif self.current_action_idx > idx:
                color = (0, 200, 0)  # completed steps

            cv2.putText(hud_frame, f"{idx}. {step}",
                        (plan_x + inner_pad, line_y),
                        font, font_scale, color, thickness, cv2.LINE_AA)
            line_y += int(20 * hud_scale)

        # ---------------- System Message (Middle section) ----------------
        sys_x = section_width + inner_pad
        sys_y = pad
        title_msg = "System message:"
        title_size = cv2.getTextSize(title_msg, font, title_font_scale, thickness)[0]
        msg_size = cv2.getTextSize(message, font, font_scale, thickness)[0] if message else (0, 0)

        # Calculate message box dimensions (1/3 of viewer width)
        msg_box_width = section_width - 2 * inner_pad
        msg_box_height = title_size[1] + msg_size[1] + 3 * inner_pad

        # Draw message box
        msg_box_tl = (sys_x, sys_y + title_size[1] + inner_pad)
        msg_box_br = (msg_box_tl[0] + msg_box_width, msg_box_tl[1] + msg_box_height)
        hud_frame = draw_box(hud_frame, msg_box_tl, msg_box_br, (0, 0, 50), 0.7)

        # Draw title + text
        cv2.putText(hud_frame, title_msg, (sys_x, sys_y + title_size[1]),
                    font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        if message:
            cv2.putText(hud_frame, message,
                        (msg_box_tl[0] + inner_pad, msg_box_tl[1] + msg_size[1] + inner_pad),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # ---------------- Held Item (Right section) ----------------
        item_x = 2 * section_width + inner_pad
        item_y = pad
        title_item = "Held item:"
        item_text = f"[{held_item}]" if held_item else "[Empty Gripper]"
        title_item_size = cv2.getTextSize(title_item, font, title_font_scale, thickness)[0]
        item_size = cv2.getTextSize(item_text, font, font_scale, thickness)[0]

        # Calculate item box dimensions (1/3 of viewer width)
        item_box_width = section_width - 2 * inner_pad
        item_box_height = title_item_size[1] + item_size[1] + 3 * inner_pad

        # Draw item box
        item_box_tl = (item_x, item_y + title_item_size[1] + inner_pad)
        item_box_br = (item_box_tl[0] + item_box_width, item_box_tl[1] + item_box_height)
        hud_frame = draw_box(hud_frame, item_box_tl, item_box_br, (50, 0, 0), 0.7)

        # Draw title + text
        cv2.putText(hud_frame, title_item, (item_x, item_y + title_item_size[1]),
                    font, title_font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(hud_frame, item_text,
                    (item_box_tl[0] + inner_pad, item_box_tl[1] + item_size[1] + inner_pad),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return hud_frame

    def display_frame(self, frame):
        import cv2
        if frame is None:
            frame = self.get_latest_robot_frame()
        cv2.imshow("first_view", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def display_augmented_frame(self):
        return self.display_frame(self.get_augmented_robot_frame(self.get_latest_robot_frame()))
