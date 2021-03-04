#lang racket/base

(require satore/clause
         satore/unification)

(define cms current-milliseconds)

;;; Stress test.
;;; There's only one predicate of two arguments.
;;; This takes basically exponential time with n.
;;; All the time is taken by clausify (safe-factoring, which includes subsumption check)
(define (stress n)
  (define pre (cms))
  (define cl1
    (time
     (clausify
      (fresh ; ensures the names are adequate variable names
       (for/list ([i n])
         `(eq #s(Var ,i) #s(Var ,(+ i 1))))))))
  (define cl2
    (time
     (clausify
      (fresh
       (for/list ([i (+ n 1)])
         `(eq #s(Var ,i) #s(Var ,(+ i 1))))))))
  (void (time (clause-subsumes cl1 cl2)))
  (void (time (clause-subsumes cl2 cl1)))
  (- (cms) pre))

;; Takes about 10s on my desktop machine for n=40 (subsumes-iter-limit=0).

(for/list ([n (in-list '(10 20 30 40))])
  (printf "n = ~a\n" n)
  (stress n))
