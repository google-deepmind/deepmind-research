#lang racket/base

(require rackunit
         "../Clause.rkt"
         (submod "../Clause.rkt" test))

;; Polarity should not count for the 'weight' cost function because otherwise it will be harder
;; to prove ~A | ~B than A | B.
(check-equal? (Clause-size (make-Clause '[p q]))
              (Clause-size (make-Clause '[(not p) (not q)])))

;; TODO: test (check-Clause-set-equivalent? Cs1 Cs2)
