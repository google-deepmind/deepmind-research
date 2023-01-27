# Pushing the Frontiers of Density Functionals by Solving the Fractional Electron Problem

This package provides a PySCF interface to the DM21 (DeepMind 21) family of
exchange-correlation functionals described in the paper ["Pushing the Frontiers
of Density Functionals by Solving the Fractional Electron
Problem"](https://doi.org/10.1126/science.abj6511).

## Installation

`pip install .` installs the interfaces to the DM21 functionals and required
dependencies. This is best done inside a
[virtual environment](https://docs.python-guide.org/dev/virtualenvs/).

Note: using PySCF 2.0 (or later) enables substantially more efficient
calculation of the local Hartree-Fock features, resulting in a large speed
increase.

### Installing directly

To install without cloning or downloading the deepmind_research repository,
execute:

```shell
python3 -m venv ~/venv/DM21
source ~/venv/DM21/bin/activate
pip install git+git://github.com/deepmind/deepmind-research.git#subdirectory=density_functional_approximation_dm21
```

### Downloading and installing from a local git repository

Alternatively, clone the deepmind_research repository and install from a local
repository:

```shell
git clone https://github.com/deepmind/deepmind-research.git
cd deepmind-research/density_functional_approximation_dm21
python3 -m venv ~/venv/DM21
source ~/venv/DM21/bin/activate
pip install .
```

The tests can be run either by running the test files directly or using
`py.test`, again from the `density_functional_approximation_dm21` subdirectory:

```shell
pip install '.[testing]'
py.test
```

## PySCF interface

A typical DFT calculation with PySCF is set up and run using:

```python
from pyscf import gto
from pyscf import dft

# Create the molecule of interest and select the basis set.
mol = gto.Mole()
mol.atom = 'Ne 0.0 0.0 0.0'
mol.basis = 'cc-pVDZ'
mol.build()

# Create a DFT solver and select the exchange-correlation functional.
mf = dft.RKS(mol)
mf.xc = 'b3lyp'

# Run the DFT calculation.
mf.kernel()
```

The DM21 functionals can be used in a very similar way, except we need to
compute local Hartree-Fock features, which does not fit in with the interface
used by conventional functionals. Instead, this package supplies a lightweight
wrapper around PySCF's numerical integration class which evaluates the
exchange-correlation energy and potential over a real-space grid. To use the
DM21 functional with PySCF, the above code needs to be changed to:

```python
import density_functional_approximation_dm21 as dm21
from pyscf import gto
from pyscf import dft

# Create the molecule of interest and select the basis set.
mol = gto.Mole()
mol.atom = 'Ne 0.0 0.0 0.0'
mol.basis = 'cc-pVDZ'
mol.build()

# Create a DFT solver and insert the DM21 functional into the solver.
mf = dft.RKS(mol)
mf._numint = dm21.NeuralNumInt(dm21.Functional.DM21)

# Run the DFT calculation.
mf.kernel()
```

i.e. instead of specifying the functional with `mf.xc = <functional name>`, the
functional is specified using `mf._numint = dm21.NeuralNumInt(<DM21
functional>)`, where `<DM21 functional>` is the corresponding member of the
`Functional` enum.

Available functionals are:

*   `DM21` - trained on molecules dataset, and fractional charge, and fractional
    spin constraints.
*   `DM21m` - trained on molecules dataset.
*   `DM21mc` - trained on molecules dataset, and fractional charge constraints.
*   `DM21mu` - trained on molecules dataset, and electron gas constraints.

Full details of the network architecture, training method and datasets used can
be found in the paper (reference below). Note that the results in our paper also
include D3 corrections, which must be
[included separately](https://pyscf.org/user/dft.html#dispersion-corrections).

### Best practices for using the neural functionals.

In this section, we suggest some tips for using the neural functionals in a way
similar to how they were used for benchmarking in the paper. The tensorflow
network that we used is running at single precision, and as such it is very hard
to converge calculations to the high convergence thresholds which are default in
pyscf. For example, the following script should allow users to run an
atomization energy calculation for methane.

```python
import density_functional_approximation_dm21 as dm21
from pyscf import gto
from pyscf import dft

# Create the molecule of interest and select the basis set.
methane = gto.Mole()
methane.atom = """H 0.000000000000 0.000000000000 0.000000000000
                  C 0.000000000000 0.000000000000 1.087900000000
                  H 1.025681956337 0.000000000000 1.450533333333
                  H -0.512840978169 0.888266630391 1.450533333333
                  H -0.512840978169 -0.888266630391 1.450533333333"""
methane.basis = 'def2-qzvp'
methane.verbose = 4
methane.build()

carbon = gto.Mole()
carbon.atom = 'C 0.0 0.0 0.0'
carbon.basis = 'def2-qzvp'
carbon.spin = 2
carbon.verbose = 4
carbon.build()

hydrogen = gto.Mole()
hydrogen.atom = 'H 0.0 0.0 0.0'
hydrogen.basis = 'def2-qzvp'
hydrogen.spin = 1
hydrogen.verbose = 4
hydrogen.build()

energies = []
for mol in [methane, carbon, hydrogen]:
  # Create a DFT solver and insert the DM21 functional into the solver.
  if mol.spin == 0:
    mf = dft.RKS(mol)
  else:
    mf = dft.UKS(mol)
  # It will make SCF faster to start close to the solution with a cheaper
  # functional.
  mf.xc = 'B3LYP'
  mf.run()
  dm0 = mf.make_rdm1()

  mf._numint = dm21.NeuralNumInt(dm21.Functional.DM21)
  # It's wise to relax convergence tolerances.
  mf.conv_tol = 1E-6
  mf.conv_tol_grad = 1E-3
  # Run the DFT calculation.
  energy = mf.kernel(dm0=dm0)
  energies.append(energy)

print({'CH4': energies[0], 'C': energies[1], 'H': energies[2]})
```

This script should produce three energies (in Hartrees) for the water molecule
and the two atoms of `{'CH4': -40.51785372584538, 'C': -37.84542045526023, 'H':
-0.5011533955627797}` , this leads to an atomization energy of 419.06 kcal/mol,
which is then corrected with the D3(BJ) correction for methane (1.20 kcal/mol)
to yield a predicted atomization energy of 420.26 kcal/mol. Comparing this to
the literature value of 420.42, leads us to deduce an error of around 0.2
kcal/mol.

It should also be noted that if a closed shell system is run unrestricted it can
give a small difference between spin densities and eigenvalues with a negligible
effect on the energy.

## Using DM21 from C++

There are two options for using the DM21 from C++.

1.  Load the SavedModel using
    [TensorFlow's C++ API](https://www.tensorflow.org/guide/saved_model#load_a_savedmodel_in_c).
    This requires a run-time dependency on the TensorFlow library.
2.  Compile the model ahead-of-time into a standalone library. This requires all
    array dimensions to be fixed at compile time, which imposes a minor
    limitation on the interface for using the functional. As the DM21
    functionals act on grid points independently, this does not restrict the
    system size.

The first option is more flexible but trickier to setup. Consequently we
demonstrate the second option: compiling the functional into a standalone
library. An example of running the DM21 functional using a standalone compiled
library is provided in `cc/dm21_aot_compiled_example.cc`. This requires a
link-time dependency on parts of the `xla_compiled_cpu_runtime_standalone`
library, which are not included in the compiled functional library. The easiest
way to build this is to use [Bazel](https://bazel.build). The first step is to
[install Bazel](https://docs.bazel.build/versions/5.3.0/install.html).
[Bazelisk](https://docs.bazel.build/versions/main/install-bazelisk.html) is
another way to install Bazel if a native installer is not available. The
following has been tested with Bazel 5.3.0. It is best to continue working
inside a virtual environment.

Assuming the above installation steps using `git clone` have been followed, and
`Bazel` has been installed, the example can be built and run using:

```
pip install -r requirements_aot_compilation.txt
bazel run --experimental_cc_shared_library --experimental_repo_remote_exec :run_dm21_aot_compiled_example
```

where the `pip install` command is only required if a fresh virtual environment
is used, and installs required prerequisites for TensorFlow. See the
[TensorFlow documentation](https://www.tensorflow.org/install/source) for more
details.

A static library, using the `cc_library` rule, can similarly be built and then
linked against from a separate program with no additional dependencies required,
shown in `cc/dm21_aot_compiled_example.cc` and the `dm21_aot_compiled_example`
build rule.

For calling from C, we recommend wrapping a C++ interface in `extern C { ... }`
to create a C API. For calling from Fortran, the C API can be used via the
Fortran 2003 ISO_C_BINDING feature.

### Detailed explanation

The supplied functionals in the `checkpoints` subdirectory are not easy to use
from C++, as they only contain operations for the forward pass through the
functional. This means they can only be used to evaluate the energy on a fixed
density. Self-consistent calculations require various gradients of the
functional, and it is easiest to create these using TensorFlow's Python API.
`NeuralNumInt` contains a method for exporting the functional and functional
derivatives. Assuming the above installation steps have been followed, a
functional and its derivatives can be exported by:

```shell
export_saved_model.py --functional=DM21 \
                      --out_dir=/path/to/export/DM21 \
                      --batch_size=1000
```

where `--out_dir` specifies the directory to save the model containing the
functional and functional derivatives to, and `--batch_size` the number of grid
points the functional will be evaluated on at a time. The functional must be the
name of a functional in the `neural_numint.Functional` enum. Note that the batch
size needs only be fixed for exporting models to be used with ahead-of-time
compilation.

The functional can now be compiled using the `saved_model_cli` tool supplied
with TensorFlow:

```shell
$ saved_model_cli aot_compile_cpu \
                  --dir /path/to/export/DM21 \
                  --output_prefix /path/to/compiled/DM21/dm21 \
                  --cpp_class dm21::functional \
                  --tag_set '' \
                  --signature_def_key default
```

The output prefix and C++ class can be arbitrarily chosen. The `tag_set` and
`signature_def_key` arguments must be as given above. This creates the following
files in the output directory:

-   dm21.h: header file defining the C++ interface to the DM21 functional.
-   dm21_makefile.inc: a snippet to be included in a Makefile for setting
    include, library and compilation flags.
-   dm21_metadata.o, dm21.o: object files for running the DM21 functional.

## Reference

If this repository is helpful for your research please cite the following
publication:

Pushing the Frontiers of Density Functionals by Solving the Fractional Electron
Problem, James Kirkpatrick, Brendan McMorrow, David H. P. Turban, Alexander L.
Gaunt, James S. Spencer, Alexander G. D. G. Matthews, Annette Obika, Louis
Thiry, Meire Fortunato, David Pfau, Lara Román Castellanos, Stig Petersen,
Alexander W. R. Nelson, Pushmeet Kohli, Paula Mori-Sánchez, Demis Hassabis, Aron
J. Cohen, Science, DOI: https://doi.org/10.1126/science.abj6511

## License

All code is made available under the Apache 2.0 License. Model parameters
(contained in the `density_functional_approximation_dm21/checkpoints/`
subdirectory) are made available under the Creative Commons Attribution 4.0
International (CC BY 4.0) License. See
https://creativecommons.org/licenses/by/4.0/legalcode for more details.

## Disclaimer

This is not an official Google product.
