#lang racket/base

(require bazaar/debug
         (except-in bazaar/order atom<=>)
         racket/file
         racket/list
         racket/random
         rackunit
         (submod satore/Clause test)
         satore/Clause
         satore/clause
         satore/misc
         satore/rewrite-tree
         satore/unification)

(*debug-level* 0)

(define-check (check-rewrite rwtree c crw)
  (define C (Clausify c))
  (define Crw (binary-rewrite-Clause rwtree C))
  (define crw-sorted (sort-clause crw))
  (unless (equal? (Clause-clause Crw) crw-sorted)
    (eprintf "c-sorted  : ~a\ncrw-sorted: ~a\n" (Clause-clause C) crw-sorted)
    (eprintf (Clause-ancestor-graph-string Crw))
    (fail-check)))

(define-simple-check (check-not-rewrite rwtree c)
  (check-rewrite rwtree c c))

;;; Self-equivalence
(let ()
  (define rwtree (make-rewrite-tree #:atom<=> atom1<=>))

  (define C1 (make-Clause (clausify '[(not (eq A B)) (eq B A)])))
  (define rls1 (Clause->rules C1 C1 #:atom<=> atom1<=>))
  (rewrite-tree-add-binary-Clause! rwtree C1 C1)
  (check-equal? (rewrite-tree-count rwtree) 2)
  ; Rewrite clause in lexicographical order
  (check-rewrite rwtree '[(eq b a)] '[(eq a b)]))

;;; Adding two converse implications
(let ()
  (define rwtree (make-rewrite-tree #:atom<=> atom1<=>))

  (define C1 (make-Clause (clausify '[(not (p A A)) (q A)])))
  (define C2 (make-Clause (clausify '[(not (q A)) (p A A)])))
  (define rls1 (Clause->rules C1 C2 #:atom<=> atom1<=>))
  (rewrite-tree-add-binary-Clause! rwtree C1 C2)
  ; This is not needed because both polarities are considered by Clause->rules:
  (rewrite-tree-add-binary-Clauses! rwtree (list C2) C1 #:rewrite? #true)
  (check-equal? (rewrite-tree-count rwtree) 2)
  (check-rewrite rwtree '[(p a a) (z c)] '[(q a) (z c)])
  (check-rewrite rwtree '[(not (p a a)) (z c)] '[(not (q a)) (z c)]))

;;; Adding rules where the converse implication is more general
(let ()
  (define rwtree (make-rewrite-tree #:atom<=> atom1<=>))

  (define Crules
    (map (compose make-Clause clausify)
         '([(not (p A A)) (q A)]
           [(not (q a)) (p a a)]
           [(not (q b)) (p b b)])))
  (for ([C (in-list (rest Crules))])
    (rewrite-tree-add-binary-Clause! rwtree C (first Crules)))
  (check-equal? (rewrite-tree-count rwtree) 4)
  (check-rewrite rwtree '[(p a a) (z c)] '[(q a) (z c)])
  (check-rewrite rwtree '[(not (p a a)) (z c)] '[(not (q a)) (z c)])
  (check-rewrite rwtree '[(p b b) (z c)] '[(q b) (z c)])
  (check-rewrite rwtree '[(not (p b b)) (z c)] '[(not (q b)) (z c)])
  (check-not-rewrite rwtree '[(p x x) (z c)]))
;;; The same with add-binary-Clauses
(let ()
  (define rwtree (make-rewrite-tree #:atom<=> atom1<=>))

  (define Crules
    (map (compose make-Clause clausify)
         '([(not (p A A)) (q A)]
           [(not (q a)) (p a a)]
           [(not (q b)) (p b b)])))
  (rewrite-tree-add-binary-Clauses! rwtree (rest Crules) (first Crules))
  (check-equal? (rewrite-tree-count rwtree) 4)
  (check-rewrite rwtree '[(p a a) (z c)] '[(q a) (z c)])
  (check-rewrite rwtree '[(not (p a a)) (z c)] '[(not (q a)) (z c)])
  (check-not-rewrite rwtree '[(p x x) (z c)]))

;;; Dynamic, non-self-converse Clauses, leading to 4 rules
(let ()
  (define rwtree (make-rewrite-tree #:atom<=> atom1<=>))

  (define C1 (make-Clause (clausify '[(not (p A B C)) (p C A B)])))
  (define C2 (make-converse-Clause C1))
  (define rls1 (Clause->rules C1 C2 #:atom<=> atom1<=>))
  (rewrite-tree-add-binary-Clause! rwtree C1 C2)
  (check-equal? (rewrite-tree-count rwtree) 4)
  (check-rewrite rwtree '[(p a b c)] '[(p a b c)])
  (check-rewrite rwtree '[(p c a b)] '[(p a b c)])
  (check-rewrite rwtree '[(p b c a)] '[(p a b c)])
  (check-rewrite rwtree '[(p b a c)] '[(p a c b)]))

;;; Some random testing to make sure atom<=> has the Groundedness property.

(define (random-atom)
  (define syms '(aaa a p q r z zzz))
  (define choices (append syms syms '(NV OV L L))) ; reduce proba of NV and OV
  (define vars '())
  (let loop ()
    (define r (random-ref choices))
    (case r
      [(NV) ; new var
       (define v (new-Var))
       (set! vars (cons v vars))
       v]
      [(OV) (if (empty? vars) (loop) (random-ref vars))] ; old vars
      [(L) (cons (random-ref syms) ; first element must be a symbol
                 (build-list (random 4) (λ (i) (loop))))]
      [else r])))

(define random-atom-bank
  (remove-duplicates (build-list 1000 (λ _ (random-atom)))))
(debug-vars (length random-atom-bank))

(define (check-groundedness atom<=> lita litb)
  (define from<=>to (atom<=> lita litb))
  ; no point in testing groundedness if we don't have from literal< to
  (assert (order<? from<=>to) lita litb from<=>to)
  (define vs (vars (list lita litb)))
  (define s (make-subst))
  (for ([v (in-list vs)])
    (subst-set!/name s v (random-ref random-atom-bank)))
  (define lita2 (substitute lita s))
  (define litb2 (substitute litb s))
  (check-equal? (atom<=> lita2 litb2) '<))

(for ([i 10000])
  (apply check-groundedness atom1<=> (Varify '[(eq A B) (eq A (mul B a))]))
  (apply check-groundedness atom1<=> (Varify '[(eq A B a) (eq A (mul B a))])))

; IMPORTANT CASE: Check circularity of the rules
; Imagine we have two clauses:
; c1 = p | q
; c2 = ~p | ~q
; They are converse implications.
; From c1 we can generate the rules:
; r1 = ~p → q
; r2 = ~q → p
; and from c2 we can generate:
; r3 = p → ~q
; r4 = q → ~p
; If we choose {r1, r4} or {r2, r3} we run in circles!
; Hence the valid choices are {r2, r4} and {r1, r2}
; {r2, r4} is justified by removing negations and considering 'p < 'q.
; {r1, r2} is justified by considering that the negated atoms 'weigh' more.
;
; Now if the two clauses are:
; c3 = ~p | q  with rules   p → q   and  ~q → ~p
; c4 = p | ~q  with rules  ~p → ~q  and  q → p
; Now we should choose q → p and ~q → ~p to avoid running in circles.
(for* ([lits (in-list '( (p q) ; + ((not p) (not q))
                         (p (not q))
                         ((distinct_points A B) (equal_points A B))
                         ((distinct_points A B) (not (equal_points A B)))
                         ))]
       [r1 (in-list
            (Clause->rules (Clausify lits) #false #:atom<=> atom1<=>))]
       [r2 (in-list
            (Clause->rules (Clausify (map lnot lits)) #false #:atom<=> atom1<=>))])
  ; The rules should NOT be circular!
  (check-not-equal?
   (Vars->symbols (list (rule-from-literal r1) (rule-to-literal r1)))
   (Vars->symbols (list (rule-to-literal r2) (rule-from-literal r2)))))

;; Saving and loading rules, especially with asymmetric rules.
(let ()
  (define rwtree  (make-rewrite-tree #:atom<=> atom1<=>))
  (define rwtree2 (make-rewrite-tree #:atom<=> atom1<=>))

  ;; Asymmetric rules
  (let ([Conv (Clausify '[(not (p A A)) (q A)])])
    (rewrite-tree-add-binary-Clauses! rwtree
                                      (map Clausify
                                           '([(not (q a)) (p a a)]
                                             [(not (q b)) (p b b)]
                                             [(not (q c)) (p c c)]))
                                      Conv))
  ; Self-converse
  (let ([C (Clausify '[(not (eq A B)) (eq B A)])])
    (rewrite-tree-add-binary-Clause! rwtree C C))
  ; Symmetric
  (rewrite-tree-add-binary-Clause! rwtree
                                   (Clausify '[(not (pp A A)) (qq A)])
                                   (Clausify '[(pp A A) (not (qq A))]))

  (define Crules (rewrite-tree-original-Clauses rwtree))
  (define f (make-temporary-file))
  (save-rules! rwtree #:rules-file f)
  (load-rules! rwtree2 #:rules-file f)
  (define Crules2 (rewrite-tree-original-Clauses rwtree2))
  (check-equal? (length Crules2) (length Crules))
  ;; not efficient
  (for ([C (in-list Crules)])
    (define cl (Clause-clause C))
    (check-not-false (for/or ([C2 (in-list Crules2)])
                       (Clause-equivalence? C C2))
                     cl)))

;; Tautology reduction
(let ()
  (define rwtree (make-rewrite-tree #:atom<=> atom1<=>))
  (rewrite-tree-add-binary-Clause! rwtree
                                   (make-Clause (clausify '[(not (p (p X))) (p X)]))
                                   (make-Clause (clausify '[(p (p X)) (not (p X))])))
  (check-equal? (rewrite-tree-stats rwtree)
                '((rules . 2)
                  (unit-rules . 0)
                  (binary-rules . 2)
                  (binary-rules-static . 2)
                  (binary-rules-dynamic . 0)))
  ; These should be reduced to tautologies and thus not added
  (rewrite-tree-add-binary-Clause! rwtree
                                   (make-Clause (clausify '[(not (p (p (p X)))) (p X)]))
                                   (make-Clause (clausify '[(p (p (p X))) (not (p X))])))
  (check-equal? (rewrite-tree-stats rwtree)
                '((rules . 2)
                  (unit-rules . 0)
                  (binary-rules . 2)
                  (binary-rules-static . 2)
                  (binary-rules-dynamic . 0))))

;; Tautology reduction by dynamic rule
;; Currently fails
#;
(let ()
  (define rwtree (make-rewrite-tree #:atom<=> atom1<=>))
  (define Cp (Clausify '[(not (p A B)) (p B A)]))
  (rewrite-tree-add-binary-Clause! rwtree Cp Cp)
  ; What should we do?
  ; The dynamic rule *can* reduce this to a tautology, but doesn't because
  ; it can't be ground-oriented.
  (check Clause-equivalence?
         (binary-rewrite-Clause rwtree (Clausify '[(p A B) (p B A) q]))
         (Clausify '[(p A B) q]))
  ; Same,  but after a rewrite
  (rewrite-tree-add-binary-Clause! rwtree
                                   (Clausify '[(not (p (f A) B)) (p A B)])
                                   (Clausify '[(p (f A) B) (not (p A B))]))
  (check Clause-equivalence?
         (binary-rewrite-Clause rwtree (Clausify '[(p (f A) B) (p B A) q]))
         (Clausify '[(p A B) q])))
