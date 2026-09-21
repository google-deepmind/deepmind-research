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
"""Utilities for time varying shape control."""

import copy
import enum
import random
from typing import List, Optional, NamedTuple, Tuple, Union

import dataclasses
import numpy as np
from scipy import interpolate

from fusion_tcv import named_array
from fusion_tcv import tcv_common


class Point(NamedTuple):
  """A point in r,z coordinates."""
  r: float
  z: float

  def to_polar(self) -> "PolarPoint":
    return PolarPoint(np.arctan2(self.z, self.r),
                      np.sqrt(self.r**2 + self.z**2))

  def __neg__(self):
    return Point(-self.r, -self.z)

  def __add__(self, pt_or_val: Union["Point", float]):
    if isinstance(pt_or_val, Point):
      return Point(self.r + pt_or_val.r, self.z + pt_or_val.z)
    else:
      return Point(self.r + pt_or_val, self.z + pt_or_val)

  def __sub__(self, pt_or_val: Union["Point", float]):
    if isinstance(pt_or_val, Point):
      return Point(self.r - pt_or_val.r, self.z - pt_or_val.z)
    else:
      return Point(self.r - pt_or_val, self.z - pt_or_val)

  def __mul__(self, pt_or_val: Union["Point", float]):
    if isinstance(pt_or_val, Point):
      return Point(self.r * pt_or_val.r, self.z * pt_or_val.z)
    else:
      return Point(self.r * pt_or_val, self.z * pt_or_val)

  def __truediv__(self, pt_or_val: Union["Point", float]):
    if isinstance(pt_or_val, Point):
      return Point(self.r / pt_or_val.r, self.z / pt_or_val.z)
    else:
      return Point(self.r / pt_or_val, self.z / pt_or_val)

  __div__ = __truediv__


def dist(p1: Union[Point, np.ndarray], p2: Union[Point, np.ndarray]) -> float:
  return np.hypot(*(p1 - p2))


ShapePoints = List[Point]


def to_shape_points(array: np.ndarray) -> ShapePoints:
  return [Point(r, z) for r, z in array]


def center_point(points: ShapePoints) -> Point:
  return sum(points, Point(0, 0)) / len(points)


class ShapeSide(enum.Enum):
  LEFT = 0
  RIGHT = 1
  NOSHIFT = 2


class PolarPoint(NamedTuple):
  angle: float
  dist: float

  def to_point(self) -> Point:
    return Point(np.cos(self.angle) * self.dist, np.sin(self.angle) * self.dist)


def evenly_spaced_angles(num: int):
  return np.arange(num) * 2 * np.pi / num


def angle_aligned_dists(points: np.ndarray, angles: np.ndarray) -> np.ndarray:
  """Return a new set of points along angles that intersect with the shape."""
  # TODO(tewalds): Walk the two arrays together for an O(n+m) algorithm instead
  # of the current O(n*m). This would work as long as they are both sorted
  # around the radial direction, so the next intersection will be near the last.
  return np.array([dist_angle_to_surface(points, a) for a in angles])


def angle_aligned_points(points: np.ndarray, num_points: int,
                         origin: Point) -> np.ndarray:
  """Given a set of points, return a new space centered at origin."""
  angles = evenly_spaced_angles(num_points)
  dists = angle_aligned_dists(points - origin, angles)
  return np.stack((np.cos(angles) * dists,
                   np.sin(angles) * dists), axis=-1) + origin


def dist_angle_to_surface(points: np.ndarray, angle: float) -> float:
  """Distance along a ray to the surface defined by a list of points."""
  for p1, p2 in zip(points, np.roll(points, 1, axis=0)):
    d = dist_angle_to_segment(p1, p2, angle)
    if d is not None:
      return d
  raise ValueError(f"Intersecting edge not found for angle: {angle}")


def dist_angle_to_segment(p1, p2, angle: float) -> Optional[float]:
  """Distance along a ray from the origin to a segment defined by two points."""
  x0, y0 = p1[0], p1[1]
  x1, y1 = p2[0], p2[1]
  a0, b0 = np.cos(angle), np.sin(angle)
  a1, b1 = 0, 0
  # Segment/segment algorithm inspired by https://stackoverflow.com/q/563198
  denom = (b0 - b1) * (x0 - x1) - (y0 - y1) * (a0 - a1)
  if denom == 0:
    return None  # Angle parallel to the segment, so can't intersect.
  xy = (a0 * (y1 - b1) + a1 * (b0 - y1) + x1 * (b1 - b0)) / denom
  eps = 0.00001  # Allow intersecting slightly beyond the endpoints.
  if -eps <= xy <= 1 + eps:  # Check it hit the segment, not just the line.
    ab = (y1 * (x0 - a1) + b1 * (x1 - x0) + y0 * (a1 - x1)) / denom
    if ab > 0:  # Otherwise it hit in the reverse direction.
      # If ab <= 1 then it's within the segment defined above, but given it's
      # a unit vector with one end at the origin this tells us the distance to
      # the intersection of an infinite ray out from the origin.
      return ab
  return None


def dist_point_to_surface(points: np.ndarray, point: np.ndarray) -> float:
  """Distance from a point to the surface defined by a list of points."""
  return min(dist_point_to_segment(p1, p2, point)
             for p1, p2 in zip(points, np.roll(points, 1, axis=0)))


def dist_point_to_segment(v: np.ndarray, w: np.ndarray, p: np.ndarray) -> float:
  """Return minimum distance between line segment vw and point p."""
  # Inspired by: https://stackoverflow.com/a/1501725
  l2 = dist(v, w)**2
  if l2 == 0.0:
    return dist(p, v)  # v == w case
  # Consider the line extending the segment, parameterized as v + t (w - v).
  # We find projection of point p onto the line.
  # It falls where t = [(p-v) . (w-v)] / |w-v|^2
  # We clamp t from [0,1] to handle points outside the segment vw.
  t = max(0, min(1, np.dot(p - v, w - v) / l2))
  projection = v + t * (w - v)  # Projection falls on the segment
  return dist(p, projection)


def sort_by_angle(points: ShapePoints) -> ShapePoints:
  center = sum(points, Point(0, 0)) / len(points)
  return sorted(points, key=lambda p: (p - center).to_polar().angle)


def spline_interpolate_points(
    points: ShapePoints, num_points: int,
    x_points: Optional[ShapePoints] = None) -> ShapePoints:
  """Interpolate along a spline to give a smooth evenly spaced shape."""
  ends = []
  if x_points:
    # Find the shape points that must allow sharp corners.
    for xp in x_points:
      for i, p in enumerate(points):
        if np.hypot(*(p - xp)) < 0.01:
          ends.append(i)

  if not ends:
    # No x-points forcing sharp corners, so use a periodic spline.
    tck, _ = interpolate.splprep(np.array(points + [points[0]]).T, s=0, per=1)
    unew = np.arange(num_points) / num_points
    out = interpolate.splev(unew, tck)
    assert len(out[0]) == num_points
    return sort_by_angle(to_shape_points(np.array(out).T))

  # Generate a spline with an shape==x-point at each end.
  new_pts = []
  for i, j in zip(ends, ends[1:] + [ends[0]]):
    pts = points[i:j+1] if i < j else points[i:] + points[:j+1]
    num_segment_points = np.round((len(pts) - 1) / len(points) * num_points)
    unew = np.arange(num_segment_points + 1) / num_segment_points
    tck, _ = interpolate.splprep(np.array(pts).T, s=0)
    out = interpolate.splev(unew, tck)
    new_pts += to_shape_points(np.array(out).T)[:-1]
  if len(new_pts) != num_points:
    raise AssertionError(
        f"Generated the wrong number of points: {len(new_pts)} != {num_points}")
  return sort_by_angle(new_pts)


@dataclasses.dataclass
class ParametrizedShape:
  """Describes a target shape from the parameter set."""
  r0: float  # Where to put the center along the radial axis.
  z0: float  # Where to put the center along the vertical axis.
  kappa: float  # Elongation of the shape. (0.8, 3)
  delta: float  # Triangulation of the shape. (-1, 1)
  radius: float  # Radius of the shape (0.22, 2.58)
  lambda_: float  # Squareness of the shape. Recommend (0, 0)
  side: ShapeSide  # Whether and which side to shift the shape to.

  @classmethod
  def uniform_random_shape(
      cls,
      r_bounds=(0.8, 0.9),
      z_bounds=(0, 0.2),
      kappa_bounds=(1.0, 1.8),  # elongation
      delta_bounds=(-0.5, 0.6),  # triangulation
      radius_bounds=(tcv_common.LIMITER_WIDTH / 2 - 0.04,
                     tcv_common.LIMITER_WIDTH / 2),
      lambda_bounds=(0, 0),  # squareness
      side=(ShapeSide.LEFT, ShapeSide.RIGHT)):
    """Return a random shape."""
    return cls(
        r0=np.random.uniform(*r_bounds),
        z0=np.random.uniform(*z_bounds),
        kappa=np.random.uniform(*kappa_bounds),
        delta=np.random.uniform(*delta_bounds),
        radius=np.random.uniform(*radius_bounds),
        lambda_=np.random.uniform(*lambda_bounds),
        side=side if isinstance(side, ShapeSide) else random.choice(side))

  def gen_points(self, num_points: int) -> Tuple[ShapePoints, Point]:
    """Generates a set of shape points, return (points, modified (r0, z0))."""
    r0 = self.r0
    z0 = self.z0
    num_warped_points = 32
    points = np.zeros((num_warped_points, 2))
    theta = evenly_spaced_angles(num_warped_points)
    points[:, 0] = r0 + self.radius * np.cos(theta + self.delta * np.sin(theta)
                                             - self.lambda_ * np.sin(2 * theta))
    points[:, 1] = z0 + self.radius * self.kappa * np.sin(theta)
    if self.side == ShapeSide.LEFT:
      wall_shift = np.min(points[:, 0]) - tcv_common.INNER_LIMITER_R
      points[:, 0] -= wall_shift
      r0 -= wall_shift
    elif self.side == ShapeSide.RIGHT:
      wall_shift = np.max(points[:, 0]) - tcv_common.OUTER_LIMITER_R
      points[:, 0] -= wall_shift
      r0 -= wall_shift
    return (spline_interpolate_points(to_shape_points(points), num_points),
            Point(r0, z0))


def trim_zero_points(points: ShapePoints) -> Optional[ShapePoints]:
  trimmed = [p for p in points if p.r != 0]
  return trimmed if trimmed else None


class Diverted(enum.Enum):
  """Whether a shape is diverted or not."""
  ANY = 0
  LIMITED = 1
  DIVERTED = 2

  @classmethod
  def from_refs(cls, references: named_array.NamedArray) -> "Diverted":
    diverted = (references["diverted", 0] == 1)
    limited = (references["limited", 0] == 1)
    if diverted and limited:
      raise ValueError("Diverted and limited doesn't make sense.")
    if diverted:
      return cls.DIVERTED
    if limited:
      return cls.LIMITED
    return cls.ANY


@dataclasses.dataclass
class Shape:
  """Full specification of a shape."""
  params: Optional[ParametrizedShape] = None
  points: Optional[ShapePoints] = None
  x_points: Optional[ShapePoints] = None
  legs: Optional[ShapePoints] = None
  diverted: Diverted = Diverted.ANY
  ip: Optional[float] = None
  limit_point: Optional[Point] = None

  @classmethod
  def from_references(cls, references: named_array.NamedArray) -> "Shape":
    """Extract a Shape from the references."""
    if any(np.any(references[name] != 0)
           for name in ("R", "Z", "kappa", "delta", "radius", "lambda")):
      params = ParametrizedShape(
          r0=references["R"][0],
          z0=references["Z"][0],
          kappa=references["kappa"][0],
          delta=references["delta"][0],
          radius=references["radius"][0],
          lambda_=references["lambda"][0],
          side=ShapeSide.NOSHIFT)
    else:
      params = None

    ip = references["Ip", 0]
    return cls(
        params,
        points=trim_zero_points(points_from_references(references, "shape1")),
        x_points=trim_zero_points(points_from_references(references,
                                                         "x_points")),
        legs=trim_zero_points(points_from_references(references, "legs")),
        limit_point=trim_zero_points(points_from_references(
            references, "limit_point")[0:1]),
        diverted=Diverted.from_refs(references),
        ip=float(ip) if ip != 0 else None)

  def gen_references(self) -> named_array.NamedArray:
    """Return the references for the parametrized shape."""
    refs = tcv_common.REF_RANGES.new_named_array()

    if self.ip is not None:
      refs["Ip", 0] = self.ip

    refs["diverted", 0] = 1 if self.diverted == Diverted.DIVERTED else 0
    refs["limited", 0] = 1 if self.diverted == Diverted.LIMITED else 0

    if self.params is not None:
      refs["R", 0] = self.params.r0
      refs["Z", 0] = self.params.z0
      refs["kappa", 0] = self.params.kappa
      refs["delta", 0] = self.params.delta
      refs["radius", 0] = self.params.radius
      refs["lambda", 0] = self.params.lambda_

    if self.points is not None:
      points = np.array(self.points)
      assert refs.names.count("shape_r") >= points.shape[0]
      refs["shape_r", :points.shape[0]] = points[:, 0]
      refs["shape_z", :points.shape[0]] = points[:, 1]

    if self.x_points is not None:
      x_points = np.array(self.x_points)
      assert refs.names.count("x_points_r") >= x_points.shape[0]
      refs["x_points_r", :x_points.shape[0]] = x_points[:, 0]
      refs["x_points_z", :x_points.shape[0]] = x_points[:, 1]

    if self.legs is not None:
      legs = np.array(self.legs)
      assert refs.names.count("legs_r") >= legs.shape[0]
      refs["legs_r", :legs.shape[0]] = legs[:, 0]
      refs["legs_z", :legs.shape[0]] = legs[:, 1]

    if self.limit_point is not None:
      refs["limit_point_r", 0] = self.limit_point.r
      refs["limit_point_z", 0] = self.limit_point.z

    return refs

  def canonical(self) -> "Shape":
    """Return a canonical shape with a fixed number of points and params."""
    num_points = tcv_common.REF_RANGES.count("shape_r")
    out = copy.deepcopy(self)

    if out.points is None:
      if out.params is None:
        raise ValueError("Can't canonicalize with no params or points.")
      out.points, center = out.params.gen_points(num_points)
      out.params.r0 = center.r
      out.params.z0 = center.z
      out.params.side = ShapeSide.NOSHIFT
    else:
      out.points = spline_interpolate_points(
          out.points, num_points, out.x_points or [])
      if out.params:
        out.params.side = ShapeSide.NOSHIFT
      else:
        # Copied from FGE. Details: https://doi.org/10.1017/S0022377815001270
        top = max(out.points, key=lambda p: p.z)
        left = min(out.points, key=lambda p: p.r)
        right = max(out.points, key=lambda p: p.r)
        bottom = min(out.points, key=lambda p: p.z)
        center = Point((left.r + right.r) / 2,
                       (top.z + bottom.z) / 2)

        radius = (right.r - left.r) / 2
        kappa = (top.z - bottom.z) / (right.r - left.r)
        delta_lower = (center.r - bottom.r) / radius  # upper triangularitiy
        delta_upper = (center.r - top.r) / radius  # lower triangularity
        delta = (delta_lower + delta_upper) / 2

        out.params = ParametrizedShape(
            r0=center.r, z0=center.z, radius=radius, kappa=kappa, delta=delta,
            lambda_=0, side=ShapeSide.NOSHIFT)

    return out


def points_from_references(
    references: named_array.NamedArray, key: str = "shape1",
    num: Optional[int] = None) -> ShapePoints:
  points = np.array([references[f"{key}_r"], references[f"{key}_z"]]).T
  if num is not None:
    points = points[:num]
  return to_shape_points(points)


@dataclasses.dataclass
class ReferenceTimeSlice:
  shape: Shape
  time: float
  hold: Optional[float] = None  # Absolute time.

  def __post_init__(self):
    if self.hold is None:
      self.hold = self.time


def canonicalize_reference_series(
    time_slices: List[ReferenceTimeSlice]) -> List[ReferenceTimeSlice]:
  """Canonicalize a full sequence of time slices."""
  outputs = []
  for ref in time_slices:
    ref_shape = ref.shape.canonical()

    prev = outputs[-1] if outputs else None
    if prev is not None and prev.hold + tcv_common.DT < ref.time:
      leg_diff = len(ref_shape.legs or []) != len(prev.shape.legs or [])
      xp_diff = len(ref_shape.x_points or []) != len(prev.shape.x_points or [])
      div_diff = ref_shape.diverted != prev.shape.diverted
      limit_diff = (
          bool(ref_shape.limit_point and ref_shape.limit_point.r > 0) !=
          bool(prev.shape.limit_point and prev.shape.limit_point.r > 0))
      if leg_diff or xp_diff or div_diff or limit_diff:
        # Try not to interpolate between a real x-point and a non-existent
        # x-point. Non-existent x-points are represented as being at the
        # origin, i.e. out to the left of the vessel, and could be interpolated
        # into place, but that's weird, so better to pop it into existence by
        # adding an extra frame one before with the new shape targets.
        # This doesn't handle the case of multiple points appearing/disappearing
        # out of order, or of one moving while the other disappears.
        outputs.append(
            ReferenceTimeSlice(
                time=ref.time - tcv_common.DT,
                shape=Shape(
                    ip=ref_shape.ip,
                    params=ref_shape.params,
                    points=ref_shape.points,
                    x_points=(prev.shape.x_points
                              if xp_diff else ref_shape.x_points),
                    legs=(prev.shape.legs if leg_diff else ref_shape.legs),
                    limit_point=(prev.shape.limit_point
                                 if limit_diff else ref_shape.limit_point),
                    diverted=(prev.shape.diverted
                              if div_diff else ref_shape.diverted))))

    outputs.append(ReferenceTimeSlice(ref_shape, ref.time, ref.hold))

  return outputs
