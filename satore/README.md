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
[configure the PATH environment variable](https://github.com/racket/racket/wiki/Configure-Command-Line-for-Racket)
to include the directory containing the `racket` and `raco` executables.

To install **satore** and its dependencies (all are Apache2/MIT licensed),
type:

```shell
raco pkg install --auto --update-deps satore
```

<!--
Cloning only the correct subdirectory of the deemind-research repository is
inconvenient, but the Racket package does that itself rather nicely.
-->

## Running Satore

First, you can check that satore is correctly installed by typing:

```shell
racket -l- satore --help
```

There are some examples in the `examples` subdirectory of the `satore`
directory, and its location can be found with

```shell
racket -e '(displayln (collection-file-path "examples" "satore"))'
```

Run a trivial example:

```shell
racket -l- satore -p path-to-satore/examples/socrates.p --proof
```

The `.p` file is assumed to be a standalone file with only comments and
`cnf(…).` lines without equality, where the logic clause must be surrounded by
parentheses. All axioms must be included. (This will likely be improved soon.)

## Results on TPTP 7.2.0 (corrected)

Corrected results on TPTP 7.2.0, on a 2-2.5GHz CPU, with a limit of 5 minutes
and 32GB per problem, either with the default settings, or with
`--no-discover-online` to disable atom rewriting:

```
┌────────────────────────────┬───────┬───────┬───────┬───────┬───────┬────────┐
│Strategy                    │   UEQ │   CNE │   CEQ │   FNE │   FEQ │    All │
│                 Class size→│  1092 │  2212 │  4436 │  1784 │  5860 │  15384 │
├────────────────────────────┼───────┼───────┼───────┼───────┼───────┼────────┤
│Satore               (all)  │   177 │  1140 │   956 │   822 │  1324 │   4419 │
│                    (proof) │   177 │  1078 │   952 │   766 │  1324 │   4297 │
│                     (sat)  │     0 │    62 │     4 │    56 │     0 │    122 │
├────────────────────────────┼───────┼───────┼───────┼───────┼───────┼────────┤
│Satore w/o atom rw   (all)  │   128 │   967 │   674 │   711 │   843 │   3323 │
│                    (proof) │   128 │   913 │   670 │   658 │   843 │   3212 │
│                     (sat)  │     0 │    54 │     4 │    53 │     0 │    111 │
└────────────────────────────┴───────┴───────┴───────┴───────┴───────┴────────┘
```
