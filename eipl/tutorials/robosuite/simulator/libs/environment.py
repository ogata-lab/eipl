#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial


tex_attrib = {
    "type": "cube",
}

mat_attrib = {
    "texrepeat": "1 1",
    "specular": "0.4",
    "shininess": "0.1",
}

redwood = CustomMaterial(
    texture="WoodRed",
    tex_name="redwood",
    mat_name="redwood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)

cube = BoxObject(
    name="cube",
    size_min=[0.020, 0.020, 0.020],
    size_max=[0.022, 0.022, 0.022],
    rgba=[1, 0, 0, 1],
    material=redwood,
)
