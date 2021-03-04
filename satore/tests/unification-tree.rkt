#lang racket/base

(require racket/list
         rackunit
         (submod satore/Clause test)
         satore/Clause
         satore/clause
         satore/unification-tree)

(let ()
  (define utree (make-unification-tree))
  (add-Clause! utree (Clausify '[(p A B) (not (q A x B))]))
  (check-Clause-set-equivalent?
   (utree-resolve+unsafe-factors utree (Clausify '[(not (p a b)) (q e x f) (q g x h) (p c d)])
                                 #:L-resolvent-pruning? #false)
   (map Clausify
        '([(p c d) (q e x f) (q g x h) (not (q a x b))]
          [(p c d) (p g h) (q e x f) (not (p a b))]
          [(p c d) (p e f) (q g x h) (not (p a b))])))

  (check-Clause-set-equivalent?
   (utree-resolve+unsafe-factors utree (Clausify '[(not (p X Y)) (r X Y Y)])
                                 #:L-resolvent-pruning? #false)
   (map Clausify '([(r A B B) (not (q A x B))]))))

(let ()
  (define utree (make-unification-tree))
  (add-Clause! utree (Clausify '[(p A b) (not (q A x c))]))
  (define C2 (Clausify '[(not (p a B)) (q d x B)]))
  (check-Clause-set-equivalent?
   (utree-resolve+unsafe-factors utree C2 #:L-resolvent-pruning? #false)
   (map Clausify '([(not (q a x c)) (q d x b)]
                   [(p d b) (not (p a c))]))))


(define (utree-remove-subsumed! utree cl)
  (define C (make-Clause cl))
  (utree-inverse-find/remove! utree C Clause-subsumes))

(define (make-utree1)
  (define utree (make-unification-tree))
  (for-each
   (λ (cl) (add-Clause! utree (make-Clause (clausify cl))))
   '([(p A) (not (q B))]
     [(q A) (r B)]
     [(p c) (r b)]))
  utree)

(let ()
  (define utree (make-utree1))
  (define removed (utree-remove-subsumed! utree (clausify '[(q X)])))
  (check-equal? (length removed) 1)
  (check-equal? (length (utree-remove-subsumed! utree (clausify '[(not (q X))]))) 1)
  (check-equal? (length (utree-remove-subsumed! utree (clausify '[(r X)]))) 1)
  (check-equal? (append* (trie-values utree)) '()))

(let ()
  (define utree (make-utree1))
  (define removed (utree-remove-subsumed! utree (clausify '[(p d)])))
  (check-equal? (length removed) 0)
  (define removed2 (utree-remove-subsumed! utree (clausify '[(p c)])))
  (check-equal? (length removed2) 1))

(let ()
  (define utree (make-utree1))
  (define removed (utree-remove-subsumed! utree (clausify '[(p X)])))
  (check-equal? (length removed) 2))

