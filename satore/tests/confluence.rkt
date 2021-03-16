#lang racket/base

(require (for-syntax racket/base syntax/parse)
         define2
         global
         racket/list
         racket/pretty
         rackunit
         satore/Clause
         satore/clause
         satore/rewrite-tree
         satore/unification)

(define-global:boolean *dynamic-ok?* #true
  "Use dynamic rules?")

(define (take-at-most l n)
  (take l (min (length l) n)))

(define (display-rwtree rwtree #:? [n-max 100])
  (define rules (rewrite-tree-rules rwtree))
  (define-values (statics dyns)
    (partition rule-static?
               (filter-not (λ (rl) (lnot? (rule-from-literal rl)))
                           rules)))
  (display-rules (take-at-most (reverse (sort-rules statics)) n-max))
  (display-rules (take-at-most (reverse (sort-rules    dyns)) n-max))
  (when (or (> (length statics) n-max) (> (length dyns) n-max))
    (displayln "(output truncated because there are too many rules)"))
  (pretty-write (rewrite-tree-stats rwtree)))

;; Adds an equivalence as rules.
;; For testing purposes.
(define (add-equiv! rwtree equiv)
  (define C (make-Clause (clausify (list (lnot (first equiv)) (second equiv)))))
  (force-add-binary-Clause! rwtree C))

(define (rewrite-literal rwt lit)
  (define-values (new-lit rls) (binary-rewrite-literal rwt lit #false))
  new-lit)

;; Given a set of implications, generate equivalence
(define (equivs->rwtree equivs
                        #:? [dynamic-ok? (*dynamic-ok?*)]
                        #:? [atom<=> (get-atom<=>)])
  (define rwt (make-rewrite-tree #:atom<=> atom<=> #:dynamic-ok? dynamic-ok?))
  (for ([equiv (in-list equivs)])
    (add-equiv! rwt equiv)
    (add-equiv! rwt (map lnot equiv)))
  rwt)

(define-syntax (test-confluence stx)
  (syntax-parse stx
    [(_ equivs expected-stats #:with rwt body ...)
     #'(let ()
         (define rwt (equivs->rwtree equivs))
         (rewrite-tree-confluence! rwt)
         (define stats (rewrite-tree-stats rwt))
         (unless (equal? stats expected-stats)
           (display-rwtree rwt))
         (check-equal? stats expected-stats)
         body ...)]
    [(_ equivs expected-stats)
     #'(test-confluence equivs expected-stats #:with _rwt)]))


(with-globals ([*bounded-confluence?* #true]
               [*dynamic-ok?* #false])
  ;; This induction does work and is not subsumed.
  ;; This is possibly the minimal induction scheme (that doesn't lead to subsumed rules).
  (test-confluence
   '([(p A (f B)) (p A B)]
     [(p C C)     d])  ; not left linear
   ; Should not produce longer rules than the parents!
   '((rules . 6)
     (unit-rules . 0)
     (binary-rules . 6)
     (binary-rules-static . 6)
     (binary-rules-dynamic . 0)))

  (test-confluence
   '([(p (f (f (f z))) (f (f (f z)))) (g (g (g b)))] ; should -> b
     [(p (f (f (f z))) (f (f (f X)))) b]
     [(p (f (f z)) (f (f X))) c]
     [(p (f z) (f X)) d]
     [(p X X) (q X)])
   '((rules . 18) ; 16 also ok
     (unit-rules . 0)
     (binary-rules . 18)
     (binary-rules-static . 18)
     (binary-rules-dynamic . 0))
   #:with rwt
   (check-equal? (rewrite-literal rwt '(p z z)) '(q z))
   (check-equal? (rewrite-literal rwt '(p (f (f (f z))) (f (f (f a))))) 'b)
   (check-equal? (rewrite-literal rwt '(p (f (f (f z))) (f (f (f z))))) 'b))

  (test-confluence
   '([(p a X) q]
     [(p X a) f]
     [(p a a) (g b)])
   '((rules . 8)
     (unit-rules . 0)
     (binary-rules . 8)
     (binary-rules-static . 8)
     (binary-rules-dynamic . 0))
   #:with rwt
   (check-equal? (rewrite-literal rwt '(p a a)) 'f)
   (check-equal? (rewrite-literal rwt '(p a b)) 'f)
   (check-equal? (rewrite-literal rwt '(g b)) 'f)
   (check-equal? (rewrite-literal rwt 'q) 'f))
  )


