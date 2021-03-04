#lang info
(define collection "satore")
(define deps '("bazaar"
               "data-lib"
               "define2"
               "global"
               "math-lib"
               "text-table"
               "base"))
(define build-deps '("rackunit-lib"
                     "scribble-lib"
                     ))
(define scribblings '(("scribblings/satore.scrbl" ())))
(define pkg-desc "First-order logic prover in CNF without equality, but with atom rewrite rules")
(define version "0.1")
(define pkg-authors '(orseau))

(define racket-launcher-names '("satore"))
(define racket-launcher-libraries '("satore.rkt"))

(define test-omit-paths '("info.rkt"
                          "last-results.rkt"
                          "parse-log.rkt"
                          "in-progress/"
                          "find-rules.rkt"
                          "print-rules.rkt"
                          "run-eprover.rkt"
                          "rules/"
                          "logs/"
                          "scribblings/"))
