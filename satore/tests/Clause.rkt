#lang racket/base

(require racket/list
         rackunit
         (submod satore/Clause test)
         satore/Clause
         satore/unification)

;; Polarity should not count for the 'weight' cost function because otherwise it will be harder
;; to prove ~A | ~B than A | B.
(check-equal? (Clause-size (make-Clause '[p q]))
              (Clause-size (make-Clause '[(not p) (not q)])))
(check-equal? (Clause-size (make-Clause '[p q]))
              (Clause-size (make-Clause '[(not p) q])))

(let ()
  (define Cs1 (map Clausify '([(p A B) (p B C) (p D E)]
                              [(q A B C) (q B A C)]
                              [(r X Y)])))
  (define Cs2 (shuffle (map (λ (C) (make-Clause (fresh (Clause-clause C)))) Cs1)))
  (check-Clause-set-equivalent? Cs1 Cs2)
  (check-Clause-set-equivalent? Cs2 Cs1))
