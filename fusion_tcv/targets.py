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
"""Reward targets that return target+actual."""

import abc
import math
from typing import List, Optional, Sequence, Tuple

import dataclasses
import numpy as np
import scipy

from fusion_tcv import fge_state
from fusion_tcv import named_array
from fusion_tcv import shape
from fusion_tcv import tcv_common


class TargetError(Exception):
  """For when a target can't be computed."""


@dataclasses.dataclass(frozen=True)
class Target:
  actual: float
  target: float

  @classmethod
  def invalid(cls):
    """This target is invalid and should be ignored. Equivalent to weight=0."""
    return cls(float("nan"), float("nan"))


class AbstractTarget(abc.ABC):
  """Measure something about the simulation, with a target and actual value."""

  @property
  def name(self) -> str:
    """Returns a name for the target."""
    return self.__class__.__name__

  @abc.abstractproperty
  def outputs(self) -> int:
    """Return the number of outputs this produces."""

  @abc.abstractmethod
  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    """Returns a list of targets."""


@dataclasses.dataclass(frozen=True)
class AbstractSingleValuePerDomainTarget(AbstractTarget):
  """Base class for single value per plasma domain targets."""
  target: Optional[Sequence[float]] = None
  indices: List[int] = dataclasses.field(default_factory=lambda: [0])

  def __post_init__(self):
    if self.indices not in ([0], [1], [0, 1]):
      raise ValueError(
          f"Invalid indices: {self.indices}, must be [0], [1] or [0, 1].")
    if self.target and len(self.target) != len(self.indices):
      raise ValueError("Wrong number of targets.")

  @property
  def outputs(self) -> int:
    return len(self.indices)

  @property
  def name(self) -> str:
    return f"{super().name}: " + ",".join(str(i) for i in self.indices)


@dataclasses.dataclass(frozen=True)
class R(AbstractSingleValuePerDomainTarget):
  """Target for R."""

  def __call__(self,
               voltages: np.ndarray,
               state: fge_state.FGEState,
               references: named_array.NamedArray) -> List[Target]:
    r_d, _, _ = state.rzip_d
    if self.target is None:
      return [Target(r_d[idx], references["R"][idx]) for idx in self.indices]
    else:
      return [Target(r_d[idx], target)
              for idx, target in zip(self.indices, self.target)]


@dataclasses.dataclass(frozen=True)
class Z(AbstractSingleValuePerDomainTarget):
  """Target for Z."""

  def __call__(self,
               voltages: np.ndarray,
               state: fge_state.FGEState,
               references: named_array.NamedArray) -> List[Target]:
    _, z_d, _ = state.rzip_d
    if self.target is None:
      return [Target(z_d[idx], references["Z"][idx]) for idx in self.indices]
    else:
      return [Target(z_d[idx], target)
              for idx, target in zip(self.indices, self.target)]


@dataclasses.dataclass(frozen=True)
class Ip(AbstractSingleValuePerDomainTarget):
  """Target for Ip."""

  def __call__(self,
               voltages: np.ndarray,
               state: fge_state.FGEState,
               references: named_array.NamedArray) -> List[Target]:
    _, _, ip_d = state.rzip_d
    if self.target is None:
      return [Target(ip_d[idx], references["Ip"][idx]) for idx in self.indices]
    else:
      return [Target(ip_d[idx], target)
              for idx, target in zip(self.indices, self.target)]


class OHCurrentsClose(AbstractTarget):
  """Target for keeping OH currents close."""

  @property
  def outputs(self) -> int:
    return 1

  def __call__(self,
               voltages: np.ndarray,
               state: fge_state.FGEState,
               references: named_array.NamedArray) -> List[Target]:
    oh_coil_currents = state.get_coil_currents_by_type("OH")
    diff = abs(oh_coil_currents[0] - oh_coil_currents[1])
    return [Target(diff, 0)]


class EFCurrents(AbstractTarget):
  """EFCurrents, useful for avoiding stuck coils."""

  @property
  def outputs(self) -> int:
    return 16

  def __call__(self,
               voltages: np.ndarray,
               state: fge_state.FGEState,
               references: named_array.NamedArray) -> List[Target]:
    currents = np.concatenate([state.get_coil_currents_by_type("E"),
                               state.get_coil_currents_by_type("F")])
    return [Target(c, 0) for c in currents]


@dataclasses.dataclass(frozen=True)
class VoltageOOB(AbstractTarget):
  """Target for how much the voltages exceed the bounds."""
  relative: bool = True

  @property
  def outputs(self) -> int:
    return tcv_common.NUM_ACTIONS

  def __call__(self,
               voltages: np.ndarray,
               state: fge_state.FGEState,
               references: named_array.NamedArray) -> List[Target]:
    bounds = tcv_common.action_spec()
    excess = (np.maximum(bounds.minimum - voltages, 0) +
              np.maximum(voltages - bounds.maximum, 0))
    if self.relative:
      excess /= (bounds.maximum - bounds.minimum)
    return [Target(v, 0) for v in excess]


@dataclasses.dataclass(frozen=True)
class ShapeElongation(AbstractSingleValuePerDomainTarget):
  """Try to keep the elongation close to the references."""

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    if self.target is not None:
      targets = self.target
    else:
      targets = references["kappa"][self.indices]
    return [Target(state.elongation[i], target)
            for i, target in zip(self.indices, targets)]


@dataclasses.dataclass(frozen=True)
class ShapeTriangularity(AbstractSingleValuePerDomainTarget):
  """Try to keep the triangularity close to the references."""

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    if self.target is not None:
      targets = self.target
    else:
      targets = references["delta"][self.indices]
    return [Target(state.triangularity[i], target)
            for i, target in zip(self.indices, targets)]


@dataclasses.dataclass(frozen=True)
class ShapeRadius(AbstractSingleValuePerDomainTarget):
  """Try to keep the shape radius close to the references."""

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    if self.target is not None:
      targets = self.target
    else:
      targets = references["radius"][self.indices]
    return [Target(state.radius[i], target)
            for i, target in zip(self.indices, targets)]


@dataclasses.dataclass(frozen=True)
class AbstractPointsTarget(AbstractTarget):
  """Base class for shape point targets."""
  points: Optional[shape.ShapePoints] = None
  ref_name: Optional[str] = None
  num_points: Optional[int] = None

  def __post_init__(self):
    if self.points is not None:
      return
    elif self.ref_name is None:
      raise ValueError("Must specify points or ref_name")
    else:
      ref_name = f"{self.ref_name}_r"
      if ref_name not in tcv_common.REF_RANGES:
        raise ValueError(f"{self.ref_name} is invalid.")
      elif (self.num_points is not None and
            self.num_points > tcv_common.REF_RANGES.count(ref_name)):
        raise ValueError(
            (f"Requesting more points ({self.num_points}) than {self.ref_name} "
             "provides."))

  @property
  def outputs(self) -> int:
    return len(self.points) if self.points is not None else self.num_points

  def _target_points(
      self, references: named_array.NamedArray) -> shape.ShapePoints:
    if self.points is not None:
      return self.points
    else:
      return shape.points_from_references(
          references, self.ref_name, self.num_points)


def splined_lcfs_points(
    state: fge_state.FGEState,
    num_points: int,
    domain: int = 0) -> shape.ShapePoints:
  """Return a smooth lcfs, cleaning FGE x-point artifacts."""
  points = state.get_lcfs_points(domain)
  x_point = (shape.Point(*state.limit_point_d[domain])
             if state.is_diverted_d[domain] else None)

  if x_point is not None:
    x_points = [x_point]
    # Drop points near the x-point due to noise in the shape projection near
    # the x-point.
    points = [p for p in points if shape.dist(p, x_point) > 0.1]
    points.append(x_point)
    points = shape.sort_by_angle(points)
  else:
    x_points = []
  return shape.spline_interpolate_points(points, num_points, x_points)


@dataclasses.dataclass(frozen=True)
class ShapeLCFSDistance(AbstractPointsTarget):
  """Try to keep the shape close to the references.

  Check the distance from the target shape points to the smooth LCFS.
  """
  ref_name: str = dataclasses.field(default="shape", init=False)
  domain: int = dataclasses.field(default=0, init=False)

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    lcfs = splined_lcfs_points(state, 90, self.domain)
    outputs = []
    for p in self._target_points(references):
      if p.r == 0:  # For invalid/changing number of points.
        outputs.append(Target.invalid())
        continue
      dist = shape.dist_point_to_surface(np.array(lcfs), np.array(p))
      outputs.append(Target(dist, 0))
    return outputs


def flux_at_points(state: fge_state.FGEState, points: np.ndarray) -> np.ndarray:
  """Return the normalized interpolated flux values at a set of points."""
  # Normalized flux such that the LCFS has a value of 1, 0 in the middle,
  # and bigger than 1 farther out.
  normalized_flux = (  # (LY.Fx - LY.FA) / (LY.FB - LY.FA)
      (state.flux - state.magnetic_axis_flux_strength) /
      (state.lcfs_flux_strength - state.magnetic_axis_flux_strength)).T
  smooth_flux = scipy.interpolate.RectBivariateSpline(
      np.squeeze(state.r_coordinates),
      np.squeeze(state.z_coordinates),
      normalized_flux)
  return smooth_flux(points[:, 0], points[:, 1], grid=False)


@dataclasses.dataclass(frozen=True)
class ShapeNormalizedLCFSFlux(AbstractPointsTarget):
  """Try to keep the shape close to the references using flux.

  Check the normalized flux values at points along the target shape. This works
  in flux space, not linear distance, so may encourage smaller plasmas than the
  distance based shape rewards.
  """
  ref_name: str = dataclasses.field(default="shape1", init=False)

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    outputs = []
    for p in self._target_points(references):
      if p.r == 0:  # For invalid/changing number of points.
        outputs.append(Target.invalid())
      else:
        outputs.append(Target(
            flux_at_points(state, np.array([p]))[0], 1))
    return outputs


@dataclasses.dataclass(frozen=True)
class LegsNormalizedFlux(ShapeNormalizedLCFSFlux):
  """Try to keep the legs references close to the LCFS."""
  ref_name: str = dataclasses.field(default="legs", init=False)


@dataclasses.dataclass(frozen=True)
class AbstractXPointTarget(AbstractPointsTarget):
  """Base class for x-point targets."""
  ref_name: str = dataclasses.field(default="x_points", init=False)


@dataclasses.dataclass(frozen=True)
class XPointFluxGradient(AbstractXPointTarget):
  """Keep target points as an X point by attempting 0 flux gradient."""

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    eps = 0.01
    targets = []
    for point in self._target_points(references):
      if point.r == 0:  # For invalid/changing number of points.
        targets.append(Target.invalid())
        continue
      diff_points = np.array([
          [point.r - eps, point.z],
          [point.r + eps, point.z],
          [point.r, point.z - eps],
          [point.r, point.z + eps],
      ])
      flux_values = flux_at_points(state, diff_points)
      diff = ((np.abs(flux_values[0] - flux_values[1]) / (2 * eps)) +
              (np.abs(flux_values[2] - flux_values[3]) / (2 * eps)))
      targets.append(Target(diff, 0))
    return targets


def _dist(p1: shape.Point, p2: shape.Point):
  return math.hypot(p1.r - p2.r, p1.z - p2.z)


def _min_dist(pt: shape.Point, points: shape.ShapePoints,
              min_dist: float) -> Tuple[Optional[int], float]:
  index = None
  for i, point in enumerate(points):
    dist = _dist(pt, point)
    if dist < min_dist:
      index = i
      min_dist = dist
  return index, min_dist


@dataclasses.dataclass(frozen=True)
class XPointDistance(AbstractXPointTarget):
  """Keep target points as an X point by attempting to minimize distance.

  This assigns the x-points to targets without replacement. The first target
  will get the distance to the nearest x-point. The second target will get the
  closest, but ignoring the one assigned to the first target point. If none are
  within `max_dist`, then no x-point is assigned and that distance will be
  returned.

  It may be worth switching to a fancier algorithm that tries to minimize the
  total distance between targets and x-points, but that's slower, and we may
  actually care about some x-points more (eg a diverted point is more
  important than one farther away).
  """
  max_dist: float = 0.2

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    x_points = state.x_points
    targets = []
    for target_point in self._target_points(references):
      if target_point.r == 0:  # For invalid/changing number of points.
        targets.append(Target.invalid())
        continue
      index, min_dist = _min_dist(target_point, x_points, self.max_dist)
      if index is not None:
        x_points.pop(index)
      targets.append(Target(min_dist, 0))
    return targets


@dataclasses.dataclass(frozen=True)
class XPointFar(AbstractXPointTarget):
  """Keep extraneous x-points far away from the LCFS.

  Returns the distance from the LCFS to any true x-point that is far from a
  target x-point.

  This assigns the x-points to targets without replacement. The first target
  will get the distance to the nearest x-point. The second target will get the
  closest, but ignoring the one assigned to the first target point. If none are
  within `max_dist`, then no x-point is assigned and that distance will be
  returned.

  It may be worth switching to a fancier algorithm that tries to minimize the
  total distance between targets and x-points, but that's slower, and we may
  actually care about some x-points more (eg a diverted point is more
  important than one farther away).
  """
  max_dist: float = 0.2
  domain: int = 0
  diverted: Optional[shape.Diverted] = None

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    if self.diverted is not None:
      target = self.diverted
    else:
      target = shape.Diverted.from_refs(references)
    if target == shape.Diverted.ANY:
      return []  # Don't care.

    x_points = state.x_points

    # Filter out x-points that are near target x-points.
    for target_point in self._target_points(references):
      if target_point.r == 0:  # For invalid/changing number of points.
        continue
      index, _ = _min_dist(target_point, x_points, self.max_dist)
      if index is not None:
        x_points.pop(index)
    if not x_points:
      return [Target(100, 0)]  # No x-point gives full reward, not weight=0.

    lcfs = state.get_lcfs_points(self.domain)
    return [Target(shape.dist_point_to_surface(np.array(lcfs), np.array(p)), 0)
            for p in x_points]


@dataclasses.dataclass(frozen=True)
class XPointNormalizedFlux(AbstractXPointTarget):
  """Keep the actual X points close to the LCFS.

  Choose the x-points based on their distance to the target x-points.
  """
  max_dist: float = 0.2
  diverted: Optional[shape.Diverted] = None

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    if self.diverted is not None:
      diverted = self.diverted
    else:
      diverted = shape.Diverted.from_refs(references)

    x_points = state.x_points
    fluxes = list(flux_at_points(state, np.array(x_points).reshape((-1, 2))))
    targets = []
    # We should probably minimize the overall distance between targets and
    # x-points, but the algorithm is complicated, so instead be greedy and
    # assume they're given in priority order, or farther apart than max_dist.
    for target_point in self._target_points(references):
      if target_point.r == 0 or diverted != shape.Diverted.DIVERTED:
        # For invalid/changing number of points.
        targets.append(Target.invalid())
        continue
      index, _ = _min_dist(target_point, x_points, self.max_dist)
      if index is not None:
        targets.append(Target(fluxes[index], 1))
        x_points.pop(index)
        fluxes.pop(index)
      else:
        targets.append(Target(0, 1))
    return targets


@dataclasses.dataclass(frozen=True)
class XPointCount(AbstractTarget):
  """Target for number of x-points. Useful to avoid more than you want."""
  target: Optional[int] = None

  @property
  def outputs(self) -> int:
    return 1

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    if self.target is not None:
      target = self.target
    else:
      target_points = shape.points_from_references(
          references, "x_points", tcv_common.REF_RANGES.count("x_points_r"))
      target = sum(1 for p in target_points if p.r != 0)
    return [Target(len(state.x_points), target)]


@dataclasses.dataclass(frozen=True)
class Diverted(AbstractTarget):
  """Target for whether the plasma is diverted by an x-point."""
  diverted: Optional[shape.Diverted] = None

  @property
  def outputs(self) -> int:
    return 1

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    if self.diverted is not None:
      target = self.diverted
    else:
      target = shape.Diverted.from_refs(references)

    actual = 1 if state.is_diverted_d[0] else 0
    if target == shape.Diverted.ANY:
      return [Target.invalid()]  # Don't care.
    elif target == shape.Diverted.DIVERTED:
      return [Target(actual, 1)]
    return [Target(actual, 0)]


@dataclasses.dataclass(frozen=True)
class LimitPoint(AbstractPointsTarget):
  """Target for where the plasma is limited, either on the wall or x-point."""
  ref_name: str = dataclasses.field(default="limit_point", init=False)
  num_points: int = dataclasses.field(default=1, init=False)
  diverted: Optional[shape.Diverted] = None
  max_dist: float = 1

  def __call__(
      self,
      voltages: np.ndarray,
      state: fge_state.FGEState,
      references: named_array.NamedArray) -> List[Target]:
    if self.diverted is not None:
      diverted_target = self.diverted
    else:
      diverted_target = shape.Diverted.from_refs(references)

    if diverted_target == shape.Diverted.ANY:
      return [Target.invalid()]

    target_point = self._target_points(references)[0]
    if target_point.r == 0:
      return [Target.invalid()]

    limit_point = shape.Point(*state.limit_point_d[0])
    dist = np.hypot(*(target_point - limit_point))

    is_diverted = state.is_diverted_d[0]
    if diverted_target == shape.Diverted.DIVERTED:
      return [Target((dist if is_diverted else self.max_dist), 0)]
    return [Target((dist if not is_diverted else self.max_dist), 0)]
