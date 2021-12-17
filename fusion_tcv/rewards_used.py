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
"""The rewards used in our experiments."""

from fusion_tcv import combiners
from fusion_tcv import rewards
from fusion_tcv import targets
from fusion_tcv import transforms


# Used in TCV#70915
FUNDAMENTAL_CAPABILITY = rewards.Reward([
    rewards.Component(
        target=targets.ShapeLCFSDistance(),
        transforms=[transforms.SoftPlus(good=0.005, bad=0.05),
                    combiners.SmoothMax(-1)]),
    rewards.Component(
        target=targets.XPointFar(),
        transforms=[transforms.Sigmoid(good=0.3, bad=0.1),
                    combiners.SmoothMax(-5)]),
    rewards.Component(
        target=targets.LimitPoint(),
        transforms=[transforms.Sigmoid(bad=0.2, good=0.1)]),
    rewards.Component(
        target=targets.XPointNormalizedFlux(num_points=1),
        transforms=[transforms.SoftPlus(bad=0.08)]),
    rewards.Component(
        target=targets.XPointDistance(num_points=1),
        transforms=[transforms.Sigmoid(good=0.01, bad=0.15)]),
    rewards.Component(
        target=targets.XPointFluxGradient(num_points=1),
        transforms=[transforms.SoftPlus(bad=3)],
        weight=0.5),
    rewards.Component(
        target=targets.Ip(),
        transforms=[transforms.SoftPlus(good=500, bad=20000)]),
    rewards.Component(
        target=targets.OHCurrentsClose(),
        transforms=[transforms.SoftPlus(good=50, bad=1050)]),
], combiners.SmoothMax(-0.5))


# Used in TCV#70920
ELONGATION = rewards.Reward([
    rewards.Component(
        target=targets.ShapeLCFSDistance(),
        transforms=[transforms.SoftPlus(good=0.003, bad=0.03),
                    combiners.SmoothMax(-1)],
        weight=3),
    rewards.Component(
        target=targets.ShapeRadius(),
        transforms=[transforms.SoftPlus(good=0.002, bad=0.02)]),
    rewards.Component(
        target=targets.ShapeElongation(),
        transforms=[transforms.SoftPlus(good=0.005, bad=0.2)]),
    rewards.Component(
        target=targets.ShapeTriangularity(),
        transforms=[transforms.SoftPlus(good=0.005, bad=0.2)]),
    rewards.Component(
        target=targets.XPointCount(),
        transforms=[transforms.Equal()]),
    rewards.Component(
        target=targets.LimitPoint(),  # Stay away from the top/baffles.
        transforms=[transforms.Sigmoid(bad=0.3, good=0.2)]),
    rewards.Component(
        target=targets.Ip(),
        transforms=[transforms.SoftPlus(good=500, bad=30000)]),
    rewards.Component(
        target=targets.VoltageOOB(),
        transforms=[combiners.Mean(), transforms.SoftPlus(bad=1)]),
    rewards.Component(
        target=targets.OHCurrentsClose(),
        transforms=[transforms.ClippedLinear(good=50, bad=1050)]),
    rewards.Component(
        name="CurrentsFarFromZero",
        target=targets.EFCurrents(),
        transforms=[transforms.Abs(),
                    transforms.SoftPlus(good=100, bad=50),
                    combiners.GeometricMean()]),
], combiner=combiners.SmoothMax(-5))


# Used in TCV#70600
ITER = rewards.Reward([
    rewards.Component(
        target=targets.ShapeLCFSDistance(),
        transforms=[transforms.SoftPlus(good=0.005, bad=0.05),
                    combiners.SmoothMax(-1)],
        weight=3),
    rewards.Component(
        target=targets.Diverted(),
        transforms=[transforms.Equal()]),
    rewards.Component(
        target=targets.XPointNormalizedFlux(num_points=2),
        transforms=[transforms.SoftPlus(bad=0.08)],
        weight=[1] * 2),
    rewards.Component(
        target=targets.XPointDistance(num_points=2),
        transforms=[transforms.Sigmoid(good=0.01, bad=0.15)],
        weight=[0.5] * 2),
    rewards.Component(
        target=targets.XPointFluxGradient(num_points=2),
        transforms=[transforms.SoftPlus(bad=3)],
        weight=[0.5] * 2),
    rewards.Component(
        target=targets.LegsNormalizedFlux(),
        transforms=[transforms.Sigmoid(good=0.1, bad=0.3),
                    combiners.SmoothMax(-5)],
        weight=2),
    rewards.Component(
        target=targets.Ip(),
        transforms=[transforms.SoftPlus(good=500, bad=20000)],
        weight=2),
    rewards.Component(
        target=targets.VoltageOOB(),
        transforms=[combiners.Mean(), transforms.SoftPlus(bad=1)]),
    rewards.Component(
        target=targets.OHCurrentsClose(),
        transforms=[transforms.ClippedLinear(good=50, bad=1050)]),
    rewards.Component(
        name="CurrentsFarFromZero",
        target=targets.EFCurrents(),
        transforms=[transforms.Abs(),
                    transforms.SoftPlus(good=100, bad=50),
                    combiners.GeometricMean()]),
], combiner=combiners.SmoothMax(-5))


# Used in TCV#70755
SNOWFLAKE = rewards.Reward([
    rewards.Component(
        target=targets.ShapeLCFSDistance(),
        transforms=[transforms.SoftPlus(good=0.005, bad=0.05),
                    combiners.SmoothMax(-1)],
        weight=3),
    rewards.Component(
        target=targets.LimitPoint(),
        transforms=[transforms.Sigmoid(bad=0.2, good=0.1)]),
    rewards.Component(
        target=targets.XPointNormalizedFlux(num_points=2),
        transforms=[transforms.SoftPlus(bad=0.08)],
        weight=[1] * 2),
    rewards.Component(
        target=targets.XPointDistance(num_points=2),
        transforms=[transforms.Sigmoid(good=0.01, bad=0.15)],
        weight=[0.5] * 2),
    rewards.Component(
        target=targets.XPointFluxGradient(num_points=2),
        transforms=[transforms.SoftPlus(bad=3)],
        weight=[0.5] * 2),
    rewards.Component(
        target=targets.LegsNormalizedFlux(),
        transforms=[transforms.Sigmoid(good=0.1, bad=0.3),
                    combiners.SmoothMax(-5)],
        weight=2),
    rewards.Component(
        target=targets.Ip(),
        transforms=[transforms.SoftPlus(good=500, bad=20000)],
        weight=2),
    rewards.Component(
        target=targets.VoltageOOB(),
        transforms=[combiners.Mean(), transforms.SoftPlus(bad=1)]),
    rewards.Component(
        target=targets.OHCurrentsClose(),
        transforms=[transforms.ClippedLinear(good=50, bad=1050)]),
    rewards.Component(
        name="CurrentsFarFromZero",
        target=targets.EFCurrents(),
        transforms=[transforms.Abs(),
                    transforms.SoftPlus(good=100, bad=50),
                    combiners.GeometricMean()]),
], combiner=combiners.SmoothMax(-5))


# Used in TCV#70457
NEGATIVE_TRIANGULARITY = rewards.Reward([
    rewards.Component(
        target=targets.ShapeLCFSDistance(),
        transforms=[transforms.SoftPlus(good=0.005, bad=0.05),
                    combiners.SmoothMax(-1)],
        weight=3),
    rewards.Component(
        target=targets.ShapeRadius(),
        transforms=[transforms.SoftPlus(bad=0.04)]),
    rewards.Component(
        target=targets.ShapeElongation(),
        transforms=[transforms.SoftPlus(bad=0.5)]),
    rewards.Component(
        target=targets.ShapeTriangularity(),
        transforms=[transforms.SoftPlus(bad=0.5)]),
    rewards.Component(
        target=targets.Diverted(),
        transforms=[transforms.Equal()]),
    rewards.Component(
        target=targets.XPointNormalizedFlux(num_points=2),
        transforms=[transforms.SoftPlus(bad=0.08)],
        weight=[1] * 2),
    rewards.Component(
        target=targets.XPointDistance(num_points=2),
        transforms=[transforms.Sigmoid(good=0.02, bad=0.15)],
        weight=[0.5] * 2),
    rewards.Component(
        target=targets.XPointFluxGradient(num_points=2),
        transforms=[transforms.SoftPlus(bad=3)],
        weight=[0.5] * 2),
    rewards.Component(
        target=targets.Ip(),
        transforms=[transforms.SoftPlus(good=500, bad=20000)],
        weight=2),
    rewards.Component(
        target=targets.VoltageOOB(),
        transforms=[combiners.Mean(), transforms.SoftPlus(bad=1)]),
    rewards.Component(
        target=targets.OHCurrentsClose(),
        transforms=[transforms.ClippedLinear(good=50, bad=1050)]),
    rewards.Component(
        name="CurrentsFarFromZero",
        target=targets.EFCurrents(),
        transforms=[transforms.Abs(),
                    transforms.SoftPlus(good=100, bad=50),
                    combiners.GeometricMean()]),
], combiner=combiners.SmoothMax(-0.5))


# Used in TCV#69545
DROPLETS = rewards.Reward([
    rewards.Component(
        target=targets.R(indices=[0, 1]),
        transforms=[transforms.Sigmoid(good=0.02, bad=0.5)],
        weight=[1, 1]),
    rewards.Component(
        target=targets.Z(indices=[0, 1]),
        transforms=[transforms.Sigmoid(good=0.02, bad=0.2)],
        weight=[1, 1]),
    rewards.Component(
        target=targets.Ip(indices=[0, 1]),
        transforms=[transforms.Sigmoid(good=2000, bad=20000)],
        weight=[1, 1]),
    rewards.Component(
        target=targets.OHCurrentsClose(),
        transforms=[transforms.ClippedLinear(good=50, bad=1050)]),
], combiner=combiners.GeometricMean())
