# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module containing model evaluation metric."""
import _thread as thread
import sys
import threading
import time
import warnings
from absl import logging
import distrax

import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing


def quit_function(fn_name):
  logging.error('%s took too long', fn_name)
  sys.stderr.flush()
  thread.interrupt_main()


def exit_after(s):
  """Use as decorator to exit function after s seconds."""
  def outer(fn):

    def inner(*args, **kwargs):
      timer = threading.Timer(s, quit_function, args=[fn.__name__])
      timer.start()
      try:
        result = fn(*args, **kwargs)
      finally:
        timer.cancel()
      return result

    return inner

  return outer


@exit_after(400)
def do_grid_search(data_x_exp, data_y, clf, parameters, cv):
  scoring_choice = 'explained_variance'
  regressor = model_selection.GridSearchCV(
      clf, parameters, cv=cv, refit=True, scoring=scoring_choice)
  regressor.fit(data_x_exp, data_y)
  return regressor


def symplectic_matrix(dim):
  """Return anti-symmetric identity matrix of given dimensionality."""
  half_dims = int(dim/2)
  eye = np.eye(half_dims)
  zeros = np.zeros([half_dims, half_dims])
  top_rows = np.concatenate([zeros, - eye], axis=1)
  bottom_rows = np.concatenate([eye, zeros], axis=1)
  return np.concatenate([top_rows, bottom_rows], axis=0)


def create_latent_mask(z0, dist_std_threshold=0.5):
  """Create mask based on informativeness of each latent dimension.

  For stochastic models those latent dimensions that are too close to the prior
  are likely to be uninformative and can be ignored.

  Args:
    z0: distribution or array of phase space
    dist_std_threshold: informative latents have average inferred stds <
      dist_std_threshold

  Returns:
    latent_mask_final: boolean mask of the same dimensionality as z0
  """
  if isinstance(z0, distrax.Normal):
    std_vals = np.mean(z0.variance(), axis=0)
  elif isinstance(z0, distrax.Distribution):
    raise NotImplementedError()
  else:
    # If the latent is deterministic, pass through all dimensions
    return np.array([True]*z0.shape[-1])

  tensor_shape = std_vals.shape
  half_dims = int(tensor_shape[-1] / 2)

  std_vals_q = std_vals[:half_dims]
  std_vals_p = std_vals[half_dims:]

  # Keep both q and corresponding p as either one is informative
  informative_latents_inds = np.array([
      x for x in range(len(std_vals_q)) if
      std_vals_q[x] < dist_std_threshold or std_vals_p[x] < dist_std_threshold
  ])

  if informative_latents_inds.shape[0] > 0:
    latent_mask_final = np.zeros_like(std_vals_q)
    latent_mask_final[informative_latents_inds] = 1
    latent_mask_final = np.concatenate([latent_mask_final, latent_mask_final])
    latent_mask_final = latent_mask_final == 1

    return latent_mask_final
  else:
    return np.array([True]*tensor_shape[-1])


def standardize_data(data):
  """Applies the sklearn standardization to the data."""
  scaler = preprocessing.StandardScaler()
  scaler.fit(data)
  return scaler.transform(data)


def find_best_polynomial(data_x, data_y, max_poly_order, rsq_threshold,
                         max_dim_n=32,
                         alpha_sweep=None,
                         max_iter=1000, cv=2):
  """Find minimal polynomial expansion that is sufficient to explain data using Lasso regression."""
  rsq = 0
  poly_order = 1

  if not np.any(alpha_sweep):
    alpha_sweep = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

  # Avoid a large polynomial expansion for large latent sizes
  if data_x.shape[-1] > max_dim_n:
    print(f'>WARNING! Data is too high dimensional at {data_x.shape[-1]}')
    print('>WARNING! Setting max_poly_order = 1')
    max_poly_order = 1

  while rsq < rsq_threshold and poly_order <= max_poly_order:
    time_start = time.perf_counter()
    poly = preprocessing.PolynomialFeatures(poly_order, include_bias=False)
    data_x_exp = poly.fit_transform(data_x)
    time_end = time.perf_counter()
    print(
        f'Took {time_end-time_start}s to create polynomial features of order '
        f'{poly_order} and size {data_x_exp.shape[1]}.')

    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      time_start = time.perf_counter()
      clf = linear_model.Lasso(
          random_state=0, max_iter=max_iter, normalize=False, warm_start=False)
      parameters = {'alpha': alpha_sweep}
      try:
        regressor = do_grid_search(data_x_exp, data_y, clf, parameters, cv)
        time_end = time.perf_counter()
        print(f'Took {time_end-time_start}s to do regression grid search.')

        # Get rsq results
        time_start = time.perf_counter()
        clf = linear_model.Lasso(
            random_state=0,
            alpha=regressor.best_params_['alpha'],
            max_iter=max_iter,
            normalize=False,
            warm_start=False)
        clf.fit(data_x_exp, data_y)
        rsq = clf.score(data_x_exp, data_y)
        time_end = time.perf_counter()
        print(f'Took {time_end-time_start}s to get rsq results.')

        old_regressor = regressor
        old_poly_order = poly_order
        old_poly = poly
        old_data_x_exp = data_x_exp
        old_rsq = rsq
        old_clf = clf
        print(f'Polynomial of order {poly_order} with '
              f' alpha={regressor.best_params_} RSQ: {rsq}')
        poly_order += 1

      except KeyboardInterrupt:
        time_end = time.perf_counter()
        print(f'Timed out after {time_end-time_start}s of doing grid search.')
        print(f'Continuing with previous poly_order={old_poly_order}...')
        regressor = old_regressor
        poly_order = old_poly_order
        poly = old_poly
        data_x_exp = old_data_x_exp
        rsq = old_rsq
        clf = old_clf
        print(f'Polynomial of order {poly_order} with '
              f' alpha={regressor.best_params_} RSQ: {rsq}')
        break

  return clf, poly, data_x_exp, rsq


def eval_monomial_grad(feature, x, w, grad_acc):
  """Accumulates gradient from polynomial features and their weights."""
  features = feature.split(' ')
  variable_indices = []
  grads = np.ones(len(features)) * w
  for i, feature in enumerate(features):
    name_and_power = feature.split('^')
    if len(name_and_power) == 1:
      name, power = name_and_power[0], 1
    else:
      name, power = name_and_power
      power = int(power)
    var_index = int(name[1:])
    variable_indices.append(var_index)
    new_prod = np.ones_like(grads) * (x[var_index] ** power)
    # This needs a special case, for situation where x[index] = 0.0
    if power == 1:
      new_prod[i] = 1.0
    else:
      new_prod[i] = power * (x[var_index] ** (power - 1))
    grads = grads * new_prod
  grad_acc[variable_indices] += grads
  return grad_acc


def compute_jacobian_manual(x, polynomial_features, weight_matrix, tolerance):
  """Computes the jacobian manually."""
  # Put together the equation for each output var
  # polynomial_features = np.array(polynomial_obj.get_feature_names())
  weight_mask = np.abs(weight_matrix) > tolerance
  weight_matrix = weight_mask * weight_matrix
  jacobians = list()
  for i in range(weight_matrix.shape[0]):
    grad_accumulator = np.zeros_like(x)
    for j, feature in enumerate(polynomial_features):
      eval_monomial_grad(feature, x, weight_matrix[i, j], grad_accumulator)
    jacobians.append(grad_accumulator)
  return np.stack(jacobians)


def calculate_jacobian_prod(jacobian, noise_eps=1e-6):
  """Calculates AA*, where A=JEJ^T and A*=JE^TJ^T, which should be I."""
  # Add noise as 0 in jacobian creates issues in calculations later
  jacobian = jacobian + noise_eps
  sym_matrix = symplectic_matrix(jacobian.shape[1])
  pred = np.matmul(jacobian, sym_matrix)
  pred = np.matmul(pred, np.transpose(jacobian))

  pred_t = np.matmul(jacobian, np.transpose(sym_matrix))
  pred_t = np.matmul(pred_t, np.transpose(jacobian))

  pred_id = np.matmul(pred, pred_t)

  return pred_id


def normalise_jacobian_prods(jacobian_preds):
  """Normalises Jacobians evaluated at various points by a constant."""
  stacked_preds = np.stack(jacobian_preds)
  # For each attempt at estimating E, get the max term, and take their average
  normalisation_factor = np.mean(np.max(np.abs(stacked_preds), axis=(1, 2)))

  if normalisation_factor != 0:
    stacked_preds = stacked_preds/normalisation_factor

  return stacked_preds


def calculate_symetric_score(
    gt_data,
    model_data,
    max_poly_order,
    max_sym_score,
    rsq_threshold,
    sym_threshold,
    evaluation_point_n,
    trajectory_n=1,
    weight_tolerance=1e-5,
    alpha_sweep=None,
    max_iter=1000,
    cv=2):
  """Finds minimal polynomial expansion to explain data using Lasso regression, gets the Jacobian of the mapping and calculates how symplectic the map is."""
  model_data = model_data[..., :gt_data.shape[0], :]

  # Fing polynomial expansion that explains enough variance in the gt data
  print('Finding best polynomial expansion...')
  time_start = time.perf_counter()
  # Clean up model data to ensure it doesn't contain NaN, infinity
  # or values too large for dtype('float32')
  model_data = np.nan_to_num(model_data)
  model_data = np.clip(model_data, -999999, 999999)

  clf, poly, model_data_exp, best_rsq = find_best_polynomial(
      model_data, gt_data, max_poly_order, rsq_threshold,
      32, alpha_sweep, max_iter, cv)
  time_end = time.perf_counter()
  print(f'Took {time_end - time_start}s to find best polynomial.')

  # Calculate Symplecticity score
  all_raw_scores = []
  features = np.array(poly.get_feature_names())

  points_per_trajectory = int(len(gt_data) / trajectory_n)
  for trajectory in range(trajectory_n):
    random_data_inds = np.random.permutation(
        range(points_per_trajectory))[:evaluation_point_n]

    jacobian_preds = []
    for point_ind in random_data_inds:
      input_data_point = model_data[points_per_trajectory * trajectory +
                                    point_ind]
      time_start = time.perf_counter()
      jacobian = compute_jacobian_manual(input_data_point, features,
                                         clf.coef_, weight_tolerance)
      pred = calculate_jacobian_prod(jacobian)
      jacobian_preds.append(pred)
      time_end = time.perf_counter()
      print(f'Took {time_end - time_start}s to evaluate jacobian '
            f'around point {point_ind}.')

    # Normalise
    normalised_jacobian_preds = normalise_jacobian_prods(jacobian_preds)
    # The score is measured as the deviation from I
    identity = np.eye(normalised_jacobian_preds.shape[-1])
    scores = np.mean(np.power(normalised_jacobian_preds - identity, 2),
                     axis=(1, 2))
    all_raw_scores.append(scores)

  sym_score = np.min([np.mean(all_raw_scores), max_sym_score])
  # Calculate final SyMetric score
  if best_rsq > rsq_threshold and sym_score < sym_threshold:
    sy_metric = 1.0
  else:
    sy_metric = 0.0

  results = {
      'poly_exp_order': poly.get_params()['degree'],
      'rsq': best_rsq,
      'sym': sym_score,
      'SyMetric': sy_metric,
  }
  with np.printoptions(precision=4, suppress=True):
    print(f'----------------FINAL RESULTS FOR {trajectory_n} '
          'TRAJECTORIES------------------')
    print(f'BEST POLYNOMIAL EXPANSION ORDER: {results["poly_exp_order"]}')
    print(f'BEST RSQ (1-best): {results["rsq"]}')
    print(f'SYMPLECTICITY SCORE AROUND ALL POINTS AND ALL '
          f'TRAJECTORIES (0-best): {sym_score}')
    print(f'SyMETRIC SCORE: {sy_metric}')
    print(f'----------------FINAL RESULTS FOR {trajectory_n} '
          f'TRAJECTORIES------------------')
  return results, clf, poly, model_data_exp
