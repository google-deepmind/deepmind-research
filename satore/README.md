# Satore: First-order logic saturation with atom rewriting

Satore is a first-order logic resolution based theorem prover in CNF without
equality, but with atom rewrite rules. New rewrite rules can be
discovered during the proof search, potentially reducing exponentially the
search space.

Satore stands for Saturation with Atom Rewriting.

## Installation

First, install the **Racket** programming language (Apache2/MIT):
https://download.racket-lang.org

Note for Linux users: Do read the comments on the download page.

You may need to
[configure the PATH environment variable](https://github.com/racket/racket/wiki/Set-your-PATH-environment-variable)
to include the directory containing the `racket` and `raco` executables.

To install **satore** and its dependencies (all are Apache2/MIT licensed),
in a directory of your choice, type:

```shell
git clone https://github.com/deepmind/deepmind-research/tree/master/satore
raco pkg install --auto --update-deps --link satore
```
<!--
We use git clone instead of the git facility of `raco` so that
the code is available at a location of the user's choice.
-->

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
