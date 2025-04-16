import dataclasses
import json
import uuid
from typing import Union, List

import jax.numpy as jnp
import numpy as np

def todict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, jnp.ndarray):
        return np.array(obj).tolist()
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
            for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj

@dataclasses.dataclass
class FeedbackMixin:
    @property
    def feedback_type(self):
        return self.__class__.__name__

    @property
    def feedback_necessary(self) -> bool:
        raise NotImplementedError()

    def error_message(self) -> Union[List[str], str]:
        raise NotImplementedError()

    def feedback_dict(self):
        try:
            asdict = self.asdict()
        except Exception as e:
            asdict = {"ERROR": "SERIALIZATION ERROR"}

        error_message = self.error_message()
        if isinstance(error_message, list):
            error_message = "\n".join(error_message)
        assert isinstance(error_message, str)

        #feedback_necessary = error_message is not None
        #if hasattr(self, "feedback_necessary"):
        feedback_necessary = self.feedback_necessary

        return {
            "Type": self.feedback_type,
            "Error message": error_message,
            "Full class": asdict,
            "Feedback necessary": feedback_necessary,
        }

filepath = f"/tmp/{uuid.uuid4().hex}.txt"
print(filepath)
def set_filepath(_filepath: str):
    global filepath
    filepath = _filepath

def log_feedback_to_file(feedback):
    with open(filepath, "a") as f:
        f.write(json.dumps(feedback.feedback_dict()) + "\n")

def log_scenedict_to_file(id, scenedict):
    with open(filepath, "a") as f:
        dico = {"IS_SCENEDICT": True, "scenedict_index": id, "scenedict": scenedict}
        f.write(json.dumps(dico) + "\n")

def iterator(_filepath = None, reverse=False):
    if _filepath is not None:
        do_filepath = _filepath
    else:
        do_filepath = filepath

    with open(do_filepath, "r") as f:
        lines = f.readlines()

    if reverse:
        lines = list(reversed(lines))

    for line in lines:
        yield json.loads(line)

def is_obj_FeedbackMixin(obj):
    return "Type" in obj and "Feedback necessary" in obj

def is_obj_scenedict(obj):
    return "IS_SCENEDICT" in obj and "scenedict_index" in obj

def is_plan_success(_filepath = None):
    for obj in iterator(_filepath, reverse=True):
        if is_obj_FeedbackMixin(obj) and (obj["Type"] in ["CorrectFinalState", "IncorrectFinalState", "UnsafeFinalState"]):
            return obj["Type"] == "CorrectFinalState"
    return False

def get_necessary_plan_feedback(_filepath = None):
    for obj in iterator(_filepath, reverse=True):
        if is_obj_FeedbackMixin(obj) and obj["Feedback necessary"] == True:
            return obj

def get_last_plan_feedback(_filepath = None):
    for obj in iterator(_filepath, reverse=True):
        if is_obj_FeedbackMixin(obj):
            return obj

def get_scenedicts(_filepath = None):
    for obj in iterator(_filepath, reverse=False):
        if is_obj_scenedict(obj):
            yield obj["scenedict"]

def get_scenedict_of_id(id, _filepath = None):
    for obj in iterator(_filepath, reverse=False):
        if is_obj_scenedict(obj) and obj["scenedict_index"] == id:
            return obj
    raise AssertionError("Could not find scene dict of id {id}")
