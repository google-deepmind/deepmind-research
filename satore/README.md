# Satore: First-order logic saturation with atomic rewriting

This is a first-order logic resolution based theorem prover in CNF without
equality, but with atom rewrite rules. New rewrite rules can be
discovered during the proof search, potentially reducing exponentially the
search space.

Satore stands for Saturation with Atom Rewriting.

## Installation

### Install racket (Apache2/MIT):
* Windows, MacOS X: https://download.racket-lang.org
* Ubuntu/Debian: `[sudo] apt install racket`
* Linux (other): [Download](https://download.racket-lang.org) the `.sh` and
  install it with `[sudo] sh racket-<something>.sh`

You may need to configure the PATH environment variable to include the
directory containing the `racket` and `raco` executables.
For Windows this directory should be something like
`C:>Program Files\Racket`.

### Install satore and its dependencies (all are Apache2/MIT licensed):

In a directory of your choice, type:

```shell
git clone https://github.com/deepmind/deepmind-research/tree/master/satore
raco pkg install --auto --link satore
```

## Running Satore

Run a trivial example:

```shell
racket -l- satore -p satore/examples/socrates.p --proof
```

To see the various flags:

```shell
racket -l- satore --help
```

The .p file is assumed to be a standalone file with only comments and
`cnf(…).` lines without equality, where the logic clause must be surrounded by
parentheses. All axioms must be included. (This will likely be improved soon.)

Note that `racket -l- satore` can be invoked from anywhere.
