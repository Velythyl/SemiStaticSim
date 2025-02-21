import os
import shutil
import uuid
from dataclasses import field, dataclass
from typing import Tuple, List, Union

import numpy as np
from ai2thor.util.runtime_assets import save_thor_asset_file

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from hippo.utils.selfdataclass import SelfDataclass

from hippo.utils.spatial_utils import scale_ai2thor_object


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

    _found_assetIds: Tuple[str] = tuple()
    _found_sizes: Tuple[Tuple[float,float,float]] = tuple()
    _found_scores: Tuple[float] = tuple()

    _clip_features: np.ndarray = None
    _desired_size: Tuple[float, float, float] = None

    _id: str = field(default_factory=lambda: str(uuid.uuid4().hex))

    def add_asset_info_(self, found_assets, found_sizes, found_scores):
        return self.replace(
            _found_assetIds=found_assets,
            _found_sizes=found_sizes,
            _found_scores=found_scores,
        )

    @property
    def _concrete_assetIds(self):
        return [f"{self.id}-{id}" for id in self._found_assetIds]

    @property
    def _found_size_scaling(self):
        return np.array(self._desired_size) / np.array(self._found_sizes)

    @property
    def id(self):
        return self.object_name + f"-{self._id}"

    def asholodeckdict(self):
        asdict = super().asdict()
        if isinstance(asdict["roomId"], HippoRoomPlan):
            asdict["roomId"] = asdict["roomId"].id

        asdict.update(
            {
                "material": None,
                "layer": "Procedural0",
                "kinematic": True,
            }
        )

        asdict["assetId"] = self._concrete_assetIds[0]
        asdict["id"] = self._concrete_assetIds[0]
        asdict["position"] = xyztuple2dict(asdict["position"])
        asdict["rotation"] = xyztuple2dict(asdict["rotation"])

        return asdict

    def __len__(self) -> int:
        return len(self._found_assetIds)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.replace(
                _found_assetIds=self._found_assetIds[index.start:index.stop:index.step],
                _found_sizes=self._found_sizes[index.start:index.stop:index.step],
                _found_scores=self._found_scores[index.start:index.stop:index.step],
            )

        return self.replace(
            _found_assetIds=(self._found_assetIds[index],),
            _found_sizes=(self._found_sizes[index],),
            _found_scores=(self._found_scores[index],),
        )

    def concretize(self, target_directory, objathor_asset_directory=OBJATHOR_ASSETS_DIR):
        all_selves = self

        for self in all_selves:
            assert len(self) == 1

            assetId = self._found_assetIds[0]
            scaling = self._found_size_scaling[0]
            concrete_assetId = self._concrete_assetIds[0]

            target_directory_for_this_self = os.path.join(target_directory, concrete_assetId)
            if os.path.exists(target_directory_for_this_self):
                return
            os.makedirs(target_directory_for_this_self, exist_ok=False)

            from ai2thor.util.runtime_assets import load_existing_thor_asset_file
            obj = load_existing_thor_asset_file(OBJATHOR_ASSETS_DIR, f"{assetId}/{assetId}")

            scaling = [1,scaling[1],1]
            obj = scale_ai2thor_object(obj, scaling)

            new_obj = {}
            for toplevel_k, toplevel_v in obj.items():
                if isinstance(toplevel_v, str) and assetId in toplevel_v:
                    toplevel_v = toplevel_v.replace(assetId, concrete_assetId)
                new_obj[toplevel_k] = toplevel_v

            save_path = f"{target_directory_for_this_self}/{self._concrete_assetIds[0]}.pkl.gz"
            save_thor_asset_file(new_obj, save_path)

            original_asset_dir = os.path.join(objathor_asset_directory, assetId)
            for other_file in os.listdir(original_asset_dir):
                shutil.copy(os.path.join(original_asset_dir, other_file), os.path.join(target_directory_for_this_self, other_file))

