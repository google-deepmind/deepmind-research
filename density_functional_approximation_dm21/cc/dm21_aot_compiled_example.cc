// Copyright 2021 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>

// Users should adjust header path to match generated location. The generated
// location is determined by the -output_prefix flags passed to saved_model_cli
// aot_compile_cpu.
#include "aot_compiled_dm21.h"  // generated

void run_dm21_compiled_functional() {
  // The functional class name is set by a flag passed to saved_model_cli
  // aot_compile_cpu.
  dm21::functional dm21_xc;

  // This assumes the functional was compiled using instructions given in the
  // README.md, in particular that the model was exported with a batch size of
  // 1000.  We will compute the functional just at a single point and pad the
  // rest of the batch using grid_weight = 0.

  constexpr int batch_dim = 1000;
  // See docstring for neural_numint.FunctionalInputs for descriptions on each
  // input feature.  Note that, due how the model is compiled, we don't need to
  // pass any values for the grid coordinates.
  float rho_a[6][batch_dim] = {
      {1.238148615359934e-11},   {-5.4047671667795604e-11},
      {-5.4047671667795604e-11}, {-7.2613887530595865e-12},
      {4.436179569857956e-10},   {5.972594378309204e-11},
  };
  float hfx_a[batch_dim][2] = {{-4.33218591e-13, -4.32842821e-13}};
  float grid_weights[batch_dim] = {1.7594189642339968};

  // Use same values for both alpha and beta electrons (restricted calculation)
  dm21_xc.set_arg_feed_rho_a_data(rho_a);
  dm21_xc.set_arg_feed_rho_b_data(rho_a);
  dm21_xc.set_arg_feed_hfx_a_data(hfx_a);
  dm21_xc.set_arg_feed_hfx_b_data(hfx_a);
  dm21_xc.set_arg_feed_grid_weights_data(grid_weights);

  std::puts("Running functional...");
  bool status = dm21_xc.Run();

  if (status) {
    std::puts("Successfully ran functional.");
    // Fetch results.
    // Other methods for fetching results exist which may be more convenient.
    // Please see the generated header.
    // See neural_numint.NeuralNumint._build_graph and
    // See neural_numint.NeuralNumint.eval_xc for more details.
    // XC potential at each grid point, shape (batch_dim).
    const float* vxc = dm21_xc.result_fetch_vxc_data();
    // Derivative of the energy wtih respect to the density.
    // In python, this has shape (2, batch_dim), where the zeroth component is
    // with respect to the alpha density and the first component with respect to
    // the beta density. In C++, a flat 1D array is returned.
    const float* vrho = dm21_xc.result_fetch_vrho_data();
    // Derivative of the energy wtih respect to sigma.
    // In python, this has shape (3, batch_dim), where the zeroth component is
    // with respect to the alpha spin channel, the first component with respect
    // to the spin channel and the third component with respect to the total. In
    // C++, a flat 1D array is returned.
    const float* vsigma = dm21_xc.result_fetch_vsigma_data();
    // Derivative of the energy wtih respect to tau, the kinetic energy density.
    // In python, this has shape (2, batch_dim), where the zeroth component is
    // with respect to the alpha spin channel, and the first component with
    // respect to the spin channel. In C++, a flat 1D array is returned.
    const float* vtau = dm21_xc.result_fetch_vtau_data();
    // Intermediates required for evaluating the contribution of local
    // Hartree-Fock features to the derivative of the Fock matrix. See
    // docstrings and comments in compute_hfx_density.py and neural_numint.py.
    // In python, this has shape (2, batch_dim, nomega), where nomega is the
    // number of omega values used for the Hartree-Fock kernels. The zeroth
    // component is with respect to the Hartree-Fock energy density at each grid
    // point for the alpha-spin density and the first component with respect to
    // the beta-spin density. In C++, a flat 1D array is returned.
    const float* vhf = dm21_xc.result_fetch_vhf_data();
    std::printf("vxc[0] = %.6g\n", vxc[0]);
    std::printf("vrho[0] = %.6g, %.6g\n", vrho[0], vrho[batch_dim]);
    std::printf("vsigma[0] = %.6g, %.6g %.6g\n", vsigma[0], vsigma[batch_dim],
                vsigma[2 * batch_dim]);
    std::printf("vtau[0] = %.6g, %.6g\n", vtau[0], vtau[batch_dim]);
    std::printf("vhf[0] = %.6g, %.6g\n", vhf[0], vhf[1]);
  } else {
    std::puts("Failed to run functional.");
  }
}
