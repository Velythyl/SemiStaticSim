import uuid
from dataclasses import field, dataclass
from typing import Tuple, Any, List, Union

import numpy as np

from hippo.flaxdataclass import selfdataclass, SelfDataclass
from hippo.spatial_utils import get_bounding_box
from hippo.string_utils import get_uuid


@dataclass
class _Hippo(SelfDataclass):
    def asdict(self):
        return {k:v for k,v in super().asdict().items() if not k.startswith("_")}


@dataclass
class HippoRoomPlan(_Hippo):
    id: str
    floor_type: str = "white hex tile, glossy"
    wall_type: str = "light grey drywall, smooth"
    coords: List[Tuple[float,float]] =  ((0, 0), (0, 6), (7, 6), (7, 0))

    def asprompt(self):
        # todo
        return "room name (must be unique) | floor type | wall type | coordinates of the foor corners of the room"

    def asholodeckstr(self):
        return f"{self.id} | {self.floor_type} | {self.wall_type} | {self.coords}"

def xyztuple2dict(tup):
    return {'x': tup[0], 'y': tup[1], 'z': tup[2]}

@dataclass
class HippoObjectPlan(_Hippo):
    object_name: str
    object_description: str
    roomId: Union[HippoRoomPlan, str]

    position: Tuple[float, float, float] = (0, 0, 0)
    rotation: Tuple[float, float, float] = (0, 0, 0)

    _selected_assetId: str = None
    _selected_size: Tuple[float, float, float] = None
    _selected_scores: float = None

    _found_assetIds: Tuple[str] = tuple()
    _found_sizes: Tuple[Tuple[float,float,float]] = tuple()
    _found_scores: Tuple[float] = tuple()


    _clip_features: np.ndarray = None
    _desired_size: Tuple[float, float, float] = None



    def add_asset_info_(self, found_assets, found_sizes, found_scores):
        return self.replace(
            _found_assetIds=found_assets,
            _found_sizes=found_sizes,
            _found_scores=found_scores,
            _selected_assetId=found_assets[0],
            _selected_size=found_sizes[0],
            _selected_scores=found_scores[0],
        )


   # _positional_relations: List[PositionalRelation] = field(default_factory=list)

    @property
    def _selected_size_scaling(self):
        return np.array(self._selected_size) / np.array(self._desired_size)

    @property
    def _found_size_scaling(self):
        return np.array(self._found_sizes) / np.array(self._desired_size)

    @property
    def id(self):
        return self.object_name + f"-{get_uuid(numchars=8)}"

    def asholodeckdict(self):
        asdict = super().asdict()
        if isinstance(asdict["roomId"], HippoRoomPlan):
            asdict["roomId"] = asdict["roomId"].id

        asdict.update(
            {
                "material": None,
                "layer": "Procedural0",
                "kinematic": False,
            }
        )

        asdict["assetId"] = self._selected_assetId
        asdict["size"] = self._selected_size
        asdict["id"] = self.id
        asdict["position"] = xyztuple2dict(asdict["position"])
        asdict["rotation"] = xyztuple2dict(asdict["rotation"])
        asdict["size"] = xyztuple2dict(asdict["size"])

        return asdict

    def __len__(self) -> int:
        return len(self._found_assetIds)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        return self.replace(
            selected_assetId=self._found_assetIds[index],
            selected_score=self._found_scores[index],
            selected_size=self._found_sizes[index],
        )


from enum import Enum

# class syntax

class PositionalRelationType(Enum):
    ON=1
    BESIDE=2


@dataclass
class PositionalRelation:
    type: PositionalRelationType
    obj1: HippoObjectPlan
    obj2: HippoObjectPlan

    @classmethod
    def set(cls, type, obj1, obj2):
        self = cls(type, obj1, obj2)
        obj1._positional_relation.append(self)
        obj2._positional_relation.append(self)
        return self
