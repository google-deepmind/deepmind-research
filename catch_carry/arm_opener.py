# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility for opening arms until they are not in contact with a prop."""

import contextlib

from dm_control.mujoco.wrapper import mjbindings
import numpy as np

_MAX_IK_ATTEMPTS = 100
_IK_MAX_CORRECTION_WEIGHT = 0.1
_JOINT_LIMIT_TOLERANCE = 1e-4
_GAP_TOLERANCE = 0.1


class _ArmPropContactRemover(object):
  """Helper class for removing contacts between an arm and a prop via IK."""

  def __init__(self, physics, arm_root, prop, gap):
    arm_geoms = arm_root.find_all('geom')
    self._arm_geom_ids = set(physics.bind(arm_geoms).element_id)
    arm_joints = arm_root.find_all('joint')
    self._arm_joint_ids = list(physics.bind(arm_joints).element_id)
    self._arm_qpos_indices = physics.model.jnt_qposadr[self._arm_joint_ids]
    self._arm_dof_indices = physics.model.jnt_dofadr[self._arm_joint_ids]

    self._prop_geoms = prop.find_all('geom')
    self._prop_geom_ids = set(physics.bind(self._prop_geoms).element_id)

    self._arm_joint_min = np.full(len(self._arm_joint_ids), float('-inf'),
                                  dtype=physics.model.jnt_range.dtype)
    self._arm_joint_max = np.full(len(self._arm_joint_ids), float('inf'),
                                  dtype=physics.model.jnt_range.dtype)
    for i, joint_id in enumerate(self._arm_joint_ids):
      if physics.model.jnt_limited[joint_id]:
        self._arm_joint_min[i], self._arm_joint_max[i] = (
            physics.model.jnt_range[joint_id])

    self._gap = gap

  def _contact_pair_is_relevant(self, contact):
    set1 = self._arm_geom_ids
    set2 = self._prop_geom_ids
    return ((contact.geom1 in set1 and contact.geom2 in set2) or
            (contact.geom2 in set1 and contact.geom1 in set2))

  def _forward_and_find_next_contact(self, physics):
    """Forwards the physics and finds the next contact to handle."""
    physics.forward()
    next_contact = None
    for contact in physics.data.contact:
      if (self._contact_pair_is_relevant(contact) and
          (next_contact is None or contact.dist < next_contact.dist)):
        next_contact = contact
    return next_contact

  def _remove_contact_ik_iteration(self, physics, contact):
    """Performs one linearized IK iteration to remove the specified contact."""
    if contact.geom1 in self._arm_geom_ids:
      sign = -1
      geom_id = contact.geom1
    else:
      sign = 1
      geom_id = contact.geom2

    body_id = physics.model.geom_bodyid[geom_id]
    normal = sign * contact.frame[:3]

    jac_dtype = physics.data.qpos.dtype
    jac = np.empty((6, physics.model.nv), dtype=jac_dtype)
    jac_pos, jac_rot = jac[:3], jac[3:]
    mjbindings.mjlib.mj_jacPointAxis(
        physics.model.ptr, physics.data.ptr,
        jac_pos, jac_rot,
        contact.pos + (contact.dist / 2) * normal, normal, body_id)

    # Calculate corrections w.r.t. all joints, disregarding joint limits.
    delta_xpos = normal * max(0, self._gap - contact.dist)
    jac_all_joints = jac_pos[:, self._arm_dof_indices]
    update_unfiltered = np.linalg.lstsq(
        jac_all_joints, delta_xpos, rcond=None)[0]

    # Filter out joints at limit that are corrected in the "wrong" direction.
    initial_qpos = np.array(physics.data.qpos[self._arm_qpos_indices])
    min_filter = np.logical_and(
        initial_qpos - self._arm_joint_min < _JOINT_LIMIT_TOLERANCE,
        update_unfiltered < 0)
    max_filter = np.logical_and(
        self._arm_joint_max - initial_qpos < _JOINT_LIMIT_TOLERANCE,
        update_unfiltered > 0)
    active_joints = np.where(
        np.logical_not(np.logical_or(min_filter, max_filter)))[0]

    # Calculate corrections w.r.t. valid joints only.
    active_dof_indices = self._arm_dof_indices[active_joints]
    jac_joints = jac_pos[:, active_dof_indices]
    update_filtered = np.linalg.lstsq(jac_joints, delta_xpos, rcond=None)[0]
    update_nv = np.zeros(physics.model.nv, dtype=jac_dtype)
    update_nv[active_dof_indices] = update_filtered

    # Calculate maximum correction weight that does not violate joint limits.
    weights = np.full_like(update_filtered, _IK_MAX_CORRECTION_WEIGHT)
    active_initial_qpos = initial_qpos[active_joints]
    active_joint_min = self._arm_joint_min[active_joints]
    active_joint_max = self._arm_joint_max[active_joints]
    for i in range(len(weights)):
      proposed_update = update_filtered[i]
      if proposed_update > 0:
        max_allowed_update = active_joint_max[i] - active_initial_qpos[i]
        weights[i] = min(max_allowed_update / proposed_update, weights[i])
      elif proposed_update < 0:
        min_allowed_update = active_joint_min[i] - active_initial_qpos[i]
        weights[i] = min(min_allowed_update / proposed_update, weights[i])
    weight = min(weights)

    # Integrate the correction into `qpos`.
    mjbindings.mjlib.mj_integratePos(
        physics.model.ptr, physics.data.qpos, update_nv, weight)

    # "Paranoid" clip the modified joint `qpos` to within joint limits.
    active_qpos_indices = self._arm_qpos_indices[active_joints]
    physics.data.qpos[active_qpos_indices] = np.clip(
        physics.data.qpos[active_qpos_indices],
        active_joint_min, active_joint_max)

  @contextlib.contextmanager
  def _override_margins_and_gaps(self, physics):
    """Context manager that overrides geom margins and gaps to `self._gap`."""
    prop_geom_bindings = physics.bind(self._prop_geoms)
    original_margins = np.array(prop_geom_bindings.margin)
    original_gaps = np.array(prop_geom_bindings.gap)
    prop_geom_bindings.margin = self._gap * (1 - _GAP_TOLERANCE)
    prop_geom_bindings.gap = self._gap * (1 - _GAP_TOLERANCE)
    yield
    prop_geom_bindings.margin = original_margins
    prop_geom_bindings.gap = original_gaps
    physics.forward()

  def remove_contacts(self, physics):
    with self._override_margins_and_gaps(physics):
      for _ in range(_MAX_IK_ATTEMPTS):
        contact = self._forward_and_find_next_contact(physics)
        if contact is None:
          return
        self._remove_contact_ik_iteration(physics, contact)
      contact = self._forward_and_find_next_contact(physics)
      if contact and contact.dist < 0:
        raise RuntimeError(
            'Failed to remove contact with prop after {} iterations. '
            'Final contact distance is {}.'.format(
                _MAX_IK_ATTEMPTS, contact.dist))


def open_arms_for_prop(physics, left_arm_root, right_arm_root, prop, gap):
  """Opens left and right arms so as to leave a specified gap with the prop."""
  left_arm_opener = _ArmPropContactRemover(physics, left_arm_root, prop, gap)
  left_arm_opener.remove_contacts(physics)
  right_arm_opener = _ArmPropContactRemover(physics, right_arm_root, prop, gap)
  right_arm_opener.remove_contacts(physics)
