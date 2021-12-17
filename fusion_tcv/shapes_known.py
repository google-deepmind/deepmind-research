# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A set of known shapes."""

from fusion_tcv import shape
from fusion_tcv import tcv_common


SHAPE_70166_0450 = shape.Shape(
    ip=-120000,
    params=shape.ParametrizedShape(
        r0=0.89,
        z0=0.25,
        kappa=1.4,
        delta=0.25,
        radius=0.25,
        lambda_=0,
        side=shape.ShapeSide.NOSHIFT),
    limit_point=shape.Point(tcv_common.INNER_LIMITER_R, 0.25),
    diverted=shape.Diverted.LIMITED)


SHAPE_70166_0872 = shape.Shape(
    ip=-110000,
    params=shape.ParametrizedShape(
        r0=0.8796,
        z0=0.2339,
        kappa=1.2441,
        delta=0.2567,
        radius=0.2390,
        lambda_=0,
        side=shape.ShapeSide.NOSHIFT,
    ),
    points=[  # 20 points
        shape.Point(0.6299, 0.1413),
        shape.Point(0.6481, 0.0577),
        shape.Point(0.6804, -0.0087),
        shape.Point(0.7286, -0.0513),
        shape.Point(0.7931, -0.0660),
        shape.Point(0.8709, -0.0513),
        shape.Point(0.9543, -0.0087),
        shape.Point(1.0304, 0.0577),
        shape.Point(1.0844, 0.1413),
        shape.Point(1.1040, 0.2340),
        shape.Point(1.0844, 0.3267),
        shape.Point(1.0304, 0.4103),
        shape.Point(0.9543, 0.4767),
        shape.Point(0.8709, 0.5193),
        shape.Point(0.7931, 0.5340),
        shape.Point(0.7286, 0.5193),
        shape.Point(0.6804, 0.4767),
        shape.Point(0.6481, 0.4103),
        shape.Point(0.6299, 0.3267),
        shape.Point(0.6240, 0.2340),
    ],
    limit_point=shape.Point(tcv_common.INNER_LIMITER_R, 0.2339),
    diverted=shape.Diverted.LIMITED)
