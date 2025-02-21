import copy
import dataclasses
import itertools
import random
from typing import Tuple, Dict

import numpy as np

from hippo.scenedata import HippoObjectPlan
from hippo.utils.selfdataclass import SelfDataclass


@dataclasses.dataclass
class ObjectComposer(SelfDataclass):
    target_dir: str
    objectplans: Tuple[HippoObjectPlan]
    asset_dir: str
    scene: Dict = dataclasses.field(default_factory=dict)

    @property
    def _object_indices(self):
        ret = []
        for obj in self.objectplans:
            ret.append(list(np.arange(len(obj))))
        return ret

    @property
    def _object_indices_prod(self):
        product = list(itertools.product(*self._object_indices))
        return product

    def __len__(self):
        return len(self._object_indices_prod)

    def generate_compositions_in_order(self, prod=None):
        if prod is None:
            prod = self._object_indices_prod

        for possible_scene in prod:
            new_objectplans = []
            for i, obj_i in enumerate(possible_scene):
                new_objectplans.append(self.objectplans[i][obj_i])
            yield self.replace(objectplans=tuple(new_objectplans)).get_scene()

    def sample_composition(self):
        prod = self._object_indices_prod

        randprod = []
        for obj in prod:
            random.shuffle(obj)
            randprod.append(obj)

        yield from self.generate_compositions_in_order(prod)

    def get_scene(self):
        scene = copy.deepcopy(self.scene)
        for obj in self.objectplans:

            if len(obj) > 1:
                obj = obj[0]

            obj.concretize(self.target_dir, self.asset_dir)

            # Ensure the 'objects' list exists in the scene
            if 'objects' not in scene:
                scene['objects'] = []

            new_object = obj.asholodeckdict()

            # Add the new object to the scene
            scene['objects'].append(new_object)

        return scene
